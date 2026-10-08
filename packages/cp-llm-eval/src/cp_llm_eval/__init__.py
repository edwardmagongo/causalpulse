"""Measuring how often a verified interpretation misreads its analysis.

Builds a fixed corpus of fact tables from real fits, runs `cp_llm.interpret`
over it, grades every statement with deterministic graders and a validated
judge, and reports the rates. Only `cp_cli` imports this package.
"""
