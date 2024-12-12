from typing import Optional, Dict, List
import unittest

def simplify_path(path: str) -> str:
    """Simplify a given Unix-style path."""
    parts = path.split("/")
    stack = []
    for part in parts:
        if part in ('', '.'):
            continue
        elif part == "..":
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return "/" + "/".join(stack)

def resolve_symlink(path: str, symlinkmap: Dict[str, str]) -> str:
    """Resolve symbolic links using the given symlink map."""
    visited = set()
    while path in symlinkmap:
        if path in visited:
            raise ValueError("Symbolic link loop detected")
        visited.add(path)
        path = symlinkmap[path]
    return path

def cd(current_dir: str, relative_dir: str, symlinkmap: Optional[Dict[str, str]] = None, user_home: Optional[str] = None) -> Optional[str]:
    """Simulate the Unix `cd` command."""
    if relative_dir.startswith("~") and user_home:
        relative_dir = relative_dir.replace("~", user_home, 1)
    if not relative_dir.startswith("/"):
        relative_dir = current_dir + "/" + relative_dir
    target_path = simplify_path(relative_dir)
    if target_path is None:
        return None
    if symlinkmap:
        target_path = resolve_symlink(target_path, symlinkmap)
    return target_path


### **单元测试**

class TestCdFunction(unittest.TestCase):
    def test_basic_cd(self):
        self.assertEqual(cd("/home/bugs", "."), "/home/bugs")
        self.assertEqual(cd("/home/bugs", "bunny"), "/home/bugs/bunny")
        self.assertEqual(cd("/home/bugs", "../daffy"), "/home/daffy")
        self.assertEqual(cd("/", ".."), "/")
        self.assertEqual(cd("/", "foo/bar/../../baz"), "/baz")

    def test_cd_with_symlink(self):
        symlinkmap = {
            "/home/bugs/lola": "/home/lola",
            "/foo/bar": "/abc",
            "/abc": "/bcd",
            "/bcd/baz": "/xyz",
            "/foo/bar/baz": "/xyz",
        }
        self.assertEqual(cd("/home/bugs", "lola/../basketball", symlinkmap), "/home/bugs/basketball")
        self.assertEqual(cd("/foo/bar", "baz", symlinkmap), "/xyz")
        self.assertEqual(cd("/foo/bar", ".", symlinkmap), "/abc")
        self.assertEqual(cd("/foo/bar", "baz", symlinkmap), "/xyz")
        self.assertEqual(cd("/foo/bar", "baz", symlinkmap), "/xyz")
        self.assertEqual(cd("/foo/bar", "baz", symlinkmap), "/xyz")

    def test_symlink_loop_detection(self):
        symlinkmap = {
            "/foo": "/bar",
            "/bar": "/baz",
            "/baz": "/foo",
        }
        with self.assertRaises(ValueError):
            cd("/", "foo", symlinkmap)

    def test_user_home(self):
        self.assertEqual(cd("/home/bugs", "~/projects", user_home="/home/bugs"), "/home/bugs/projects")
        self.assertEqual(cd("/home/bugs", "~", user_home="/home/bugs"), "/home/bugs")
        self.assertEqual(cd("/home/bugs", "~/../another", user_home="/home/bugs"), "/home/another")

if __name__ == "__main__":
    unittest.main()
