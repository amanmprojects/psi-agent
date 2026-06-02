from langchain.tools import tool


@tool
def bash(command: str) -> str:
    """Run a bash command and return the output.
    Args:
        command (str): The bash command to run.
    """
    import subprocess

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout
