from langchain.tools import tool


@tool
def bash(command: str) -> str:
    """Run a bash command and return the output.

    Args:
        command (str): The bash command to run.

    Returns:
        str: The output of the bash command.
    """
    import subprocess

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout


tools = [bash]
tools_by_name = {tool.name: tool for tool in tools}
