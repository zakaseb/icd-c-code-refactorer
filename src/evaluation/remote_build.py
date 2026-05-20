import paramiko
from pathlib import Path

## REMOTE HOST CONFIGURATION
REMOTE_BUILD_HOST='192.168.194.14' # CHANGE IF NEEDED
REMOTE_BUILD_USER='nvidia'
REMOTE_BUILD_PASS='nvidia'
REMOTE_BUILD_SRC=Path('C:/superloop-sw-develop-copy/Workspace/P3_MCP_Application/cmake-src')

## LOCAL HOST CONFIGURATION (UPDATE)
WORKSPACE_PATH = Path('/home/nvidia/Desktop/icd-c-code-refactorer-agentic-debug-loop/workspace/sessions/6d45581c-4719-494f-a96d-f77540409b47')

## SSH CONNECTION
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(REMOTE_BUILD_HOST, username=REMOTE_BUILD_USER, password=REMOTE_BUILD_PASS)
sftp = ssh.open_sftp()

# COPY GENERATED FILES TO WINDOWS
print("Copying modified files...")
gen_files = sorted(p for p in (WORKSPACE_PATH/"original_code").iterdir() if p.is_file() and p.suffix in ('.c', '.h'))
for gf in gen_files:
    print(gf.name)
    remote_path = f"{REMOTE_BUILD_SRC}/src/{gf.name}"
    print(remote_path)
    local_path  = str(WORKSPACE_PATH / "original_code" / gf.name)
    sftp.put(local_path, remote_path)
    print(f"  {local_path} -> {remote_path}")

# REBUILD
print("Building...")
build_arg = ""
stdin, stdout, stderr = ssh.exec_command(
    f"cd {REMOTE_BUILD_SRC} && python build.py {build_arg}"
)
for line in stdout:
    print(line.strip())
errors = stderr.read().decode()
if errors:
    print("Build errors:", errors)

ssh.close();