import errno
import threading
import time
from fuse import FUSE, Operations, LoggingMixIn

class GPTfs(LoggingMixIn, Operations):
    def __init__(self):
        self.sessions = {}  #每个session表示一次对话 
        self.mutex = threading.Lock()

    def _ensure_session_files(self, session): #如果还不存在该session，则初始化一个
        if session not in self.sessions:
            self.sessions[session] = {
                'input': b'',
                'output': b'',
                'error': b''
            }

    def readdir(self, path, fh):
        if path == "/":
            return [".", ".."] + list(self.sessions.keys())
        parts = path.strip("/").split("/")
        if len(parts) == 1 and parts[0] in self.sessions:
            return [".", "..", "input", "output", "error"]
        raise OSError(errno.ENOENT, "")

    def getattr(self, path, fh=None):
        now = int(time.time())
        parts = path.strip("/").split("/")
        if path == "/":
            return dict(st_mode=(0o40755), st_nlink=2)
        elif len(parts) == 1:
            if parts[0] in self.sessions:
                return dict(st_mode=(0o40755), st_nlink=2)
        elif len(parts) == 2:
            session, file = parts
            if session in self.sessions and file in ("input", "output", "error"):
                content = self.sessions[session][file]
                return dict(st_mode=(0o100644), st_nlink=1, st_size=len(content))
        raise OSError(errno.ENOENT, "")

    def mkdir(self, path, mode):
        session = path.strip("/")
        with self.mutex:
            if session in self.sessions:
                raise OSError(errno.EEXIST, "")
            self._ensure_session_files(session)

    def open(self, path, flags):
        return 0

    def read(self, path, size, offset, fh):
        parts = path.strip("/").split("/")
        if len(parts) == 2:
            session, file = parts
            self._ensure_session_files(session)
            content = self.sessions[session][file]
            return content[offset:offset+size]
        raise OSError(errno.ENOENT, "")

    def write(self, path, data, offset, fh):
        parts = path.strip("/").split("/")
        if len(parts) == 2:
            session, file = parts
            self._ensure_session_files(session)
            if file == "input":
                # 写入 prompt
                self.sessions[session]["input"] = data
                # 模拟 GPT 生成回复
                try:
                    prompt = data.decode()
                    fake_response = f"模拟回答：你说的是『{prompt.strip()}』。".encode()
                    self.sessions[session]["output"] = fake_response
                    self.sessions[session]["error"] = b''
                except Exception as e:
                    self.sessions[session]["output"] = b''
                    self.sessions[session]["error"] = str(e).encode()
                return len(data)
            else:
                raise OSError(errno.EPERM, "只允许写 input 文件")
        raise OSError(errno.ENOENT, "")

    def truncate(self, path, length):
        parts = path.strip("/").split("/")
        if len(parts) == 2:
            session, file = parts
            self._ensure_session_files(session)
            self.sessions[session][file] = self.sessions[session][file][:length]

    def create(self, path, mode):
        raise OSError(errno.EPERM, "不允许创建新文件")

    def unlink(self, path):
        raise OSError(errno.EPERM, "不允许删除文件")

    def rmdir(self, path):
        session = path.strip("/")
        if session in self.sessions:
            del self.sessions[session]
        else:
            raise OSError(errno.ENOENT, "")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python gptfs.py <挂载点>")
        sys.exit(1)
    mountpoint = sys.argv[1] #挂载点,也是FUSE文件系统的根目录
    FUSE(GPTfs(), mountpoint, nothreads=True, foreground=True) #将GPTfs挂载到挂载点
