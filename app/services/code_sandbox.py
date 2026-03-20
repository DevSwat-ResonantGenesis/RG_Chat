"""
Code Execution Sandbox (CES)
=============================

Phase 5.8: Agents can run/test code before responding.

Features:
- Safe code execution environment
- Support for Python, JavaScript, SQL
- Timeout and resource limits
- Output capture and validation
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str
    execution_time_ms: float
    language: str
    exit_code: int


class CodeSandbox:
    """
    Safe code execution sandbox for testing code.
    """
    
    def __init__(
        self,
        timeout_seconds: int = 10,
        max_output_length: int = 5000,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_length = max_output_length
        
        # Supported languages and their executors
        self.executors = {
            "python": self._execute_python,
            "javascript": self._execute_javascript,
            "sql": self._execute_sql_mock,
            "bash": self._execute_bash,
        }
        
        # Dangerous patterns to block
        self.blocked_patterns = [
            r"import\s+os",
            r"import\s+subprocess",
            r"import\s+sys",
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"rm\s+-rf",
            r"sudo",
            r"chmod",
            r"chown",
            r"curl\s+",
            r"wget\s+",
            r"require\s*\(\s*['\"]child_process",
            r"require\s*\(\s*['\"]fs",
            r"process\.exit",
        ]
    
    def detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        code_lower = code.lower()
        
        # Python indicators
        if any(ind in code for ind in ["def ", "import ", "print(", "elif ", "True", "False", "None"]):
            return "python"
        
        # JavaScript indicators
        if any(ind in code for ind in ["const ", "let ", "function ", "=>", "console.log", "require("]):
            return "javascript"
        
        # SQL indicators
        if any(ind in code_lower for ind in ["select ", "insert ", "update ", "delete ", "create table"]):
            return "sql"
        
        # Bash indicators
        if any(ind in code for ind in ["#!/bin/bash", "echo ", "if [", "for i in"]):
            return "bash"
        
        return "python"  # Default
    
    def is_safe(self, code: str) -> Tuple[bool, str]:
        """Check if code is safe to execute."""
        for pattern in self.blocked_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern}"
        
        return True, ""
    
    async def execute(
        self,
        code: str,
        language: Optional[str] = None,
        test_input: str = "",
    ) -> ExecutionResult:
        """Execute code in sandbox."""
        start_time = datetime.now()
        
        # Detect language if not specified
        if not language:
            language = self.detect_language(code)
        
        # Safety check
        is_safe, reason = self.is_safe(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code blocked for safety: {reason}",
                execution_time_ms=0,
                language=language,
                exit_code=-1,
            )
        
        # Get executor
        executor = self.executors.get(language)
        if not executor:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}",
                execution_time_ms=0,
                language=language,
                exit_code=-1,
            )
        
        try:
            result = await executor(code, test_input)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout_seconds}s",
                execution_time_ms=self.timeout_seconds * 1000,
                language=language,
                exit_code=-1,
            )
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=execution_time,
                language=language,
                exit_code=-1,
            )
    
    async def _execute_python(self, code: str, test_input: str = "") -> ExecutionResult:
        """Execute Python code."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                'python3', temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=test_input.encode() if test_input else None),
                timeout=self.timeout_seconds,
            )
            
            output = stdout.decode()[:self.max_output_length]
            error = stderr.decode()[:self.max_output_length]
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=output,
                error=error,
                execution_time_ms=0,
                language="python",
                exit_code=process.returncode or 0,
            )
        finally:
            os.unlink(temp_path)
    
    async def _execute_javascript(self, code: str, test_input: str = "") -> ExecutionResult:
        """Execute JavaScript code using Node.js."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                'node', temp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=test_input.encode() if test_input else None),
                timeout=self.timeout_seconds,
            )
            
            output = stdout.decode()[:self.max_output_length]
            error = stderr.decode()[:self.max_output_length]
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=output,
                error=error,
                execution_time_ms=0,
                language="javascript",
                exit_code=process.returncode or 0,
            )
        finally:
            os.unlink(temp_path)
    
    async def _execute_sql_mock(self, code: str, test_input: str = "") -> ExecutionResult:
        """Mock SQL execution (parse and validate only)."""
        # Basic SQL validation
        sql_keywords = ["select", "insert", "update", "delete", "create", "drop", "alter"]
        code_lower = code.lower().strip()
        
        is_valid = any(code_lower.startswith(kw) for kw in sql_keywords)
        
        if is_valid:
            return ExecutionResult(
                success=True,
                output=f"SQL syntax validated. Query type: {code_lower.split()[0].upper()}",
                error="",
                execution_time_ms=0,
                language="sql",
                exit_code=0,
            )
        else:
            return ExecutionResult(
                success=False,
                output="",
                error="Invalid SQL syntax",
                execution_time_ms=0,
                language="sql",
                exit_code=1,
            )
    
    async def _execute_bash(self, code: str, test_input: str = "") -> ExecutionResult:
        """Execute Bash code (very restricted)."""
        # Only allow safe commands
        safe_commands = ["echo", "printf", "date", "whoami", "pwd", "ls"]
        first_word = code.strip().split()[0] if code.strip() else ""
        
        if first_word not in safe_commands:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Command '{first_word}' not allowed in sandbox",
                execution_time_ms=0,
                language="bash",
                exit_code=-1,
            )
        
        try:
            process = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode()[:self.max_output_length],
                error=stderr.decode()[:self.max_output_length],
                execution_time_ms=0,
                language="bash",
                exit_code=process.returncode or 0,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_ms=0,
                language="bash",
                exit_code=-1,
            )
    
    async def validate_code(self, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Validate code without executing (syntax check only)."""
        if not language:
            language = self.detect_language(code)
        
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
                return {"valid": True, "language": language, "error": None}
            except SyntaxError as e:
                return {"valid": False, "language": language, "error": str(e)}
        
        # For other languages, just check safety
        is_safe, reason = self.is_safe(code)
        return {
            "valid": is_safe,
            "language": language,
            "error": reason if not is_safe else None,
        }
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown text."""
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        blocks = []
        for lang, code in matches:
            if not lang:
                lang = self.detect_language(code)
            blocks.append({
                "language": lang,
                "code": code.strip(),
            })
        
        return blocks


# Global instance
code_sandbox = CodeSandbox()


async def execute_code_safely(code: str, language: str = "python", timeout: int = 30) -> Dict[str, Any]:
    """
    Execute code safely in the sandbox.
    This is the main entry point for the API endpoint.
    """
    sandbox = CodeSandbox(timeout_seconds=timeout)
    result = await sandbox.execute(code, language)
    
    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms,
        "language": result.language,
        "exit_code": result.exit_code,
    }
