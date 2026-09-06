"""One module per subcommand.

Each exposes an `add_arguments`-style function that populates a parser and a
handler taking parsed arguments and returning an exit code. `bench` has two of
each, since `run` and `report` are separate subcommands under it. No command
module reads `sys.argv` or calls `sys.exit`, so every handler is callable from
a test without a subprocess.
"""
