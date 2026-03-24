"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.tools.tools_runtime import ToolRuntime


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
    rag_config: dict | None = None


OBSERVATION_MAX_LEN: int = 10000
OBSERVATION_HEAD_LEN: int = 5000
OBSERVATION_TAIL_LEN: int = 5000
OBSERVATION_TRUNCATED_NOTICE: str = (
    "\n<warning>\n"
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n"
    "</warning>\n"
)


def truncate_observation(text: str) -> str:
    """Truncate long observation to head + notice + elided + tail."""
    if not text or len(text) <= OBSERVATION_MAX_LEN:
        return text
    elided = len(text) - OBSERVATION_HEAD_LEN - OBSERVATION_TAIL_LEN
    return (
        text[:OBSERVATION_HEAD_LEN]
        + OBSERVATION_TRUNCATED_NOTICE
        + f"<elided>{elided} characters elided</elided>\n"
        + text[-OBSERVATION_TAIL_LEN:]
    )


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.toolruntime = ToolRuntime(rag_config=self.config.rag_config)

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        all_vars = template_vars | self.extra_template_vars | kwargs
        return Template(template, undefined=StrictUndefined).render(**all_vars)

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        output = response["content"]
        msg_kwargs = {}
        if response.get("tools") and not self._will_use_bash(response):
            msg_kwargs["tool_calls"] = response["tools"]
        if response.get("extra"):
            msg_kwargs["extra"] = response["extra"]
        self.add_message("assistant", output, **msg_kwargs)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.parse_action(response)

        last_msg = self.messages[-1] if self.messages else {}
        if last_msg.get("role") == "assistant" and last_msg.get("tool_calls") and response.get("tools"):
            tool_info = response["tools"]
            result_content = json.dumps(output) if isinstance(output, dict) else str(output)
            result_content = truncate_observation(result_content)
            self.add_message(
                "tool",
                result_content,
                tool_call_id=tool_info.get("id", ""),
                name=tool_info["function"]["name"],
            )
        else:
            output_for_render = {
                **output,
                "output": truncate_observation(output.get("output", "")),
            }
            observation = self.render_template(self.config.action_observation_template, output=output_for_render)
            self.add_message("user", observation)
        return output

    @staticmethod
    def _will_use_bash(response: dict) -> bool:
        """Return True when parse_action would execute a bash block for this response."""
        content = response.get("content", "")
        if not content:
            return False
        return len(re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)) == 1

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        content = response.get("content", "")
        actions = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL) if content else []
        if len(actions) == 1:
            return self.execute_action({"action": actions[0].strip(), **response})
        if response.get("tools"):
            from minisweagent.tools.submit import Submitted as ToolSubmitted

            try:
                result = self.toolruntime.dispatch(tool_call=response["tools"]["function"])
                self.has_finished(result)
            except ToolSubmitted as e:
                raise Submitted(str(e))
            return result
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))
