# Sandbox Agent SDK in E2B Sandbox (JavaScript)

This example shows how to run [Sandbox Agent](https://github.com/rivet-dev/sandbox-agent) inside an E2B sandbox, then connect to it using the Sandbox Agent TypeScript SDK.

## What this example does

1. Creates an E2B sandbox.
2. Installs Sandbox Agent CLI in the sandbox.
3. Installs a coding agent (`codex`, `claude`, or any `SANDBOX_AGENT` value).
4. Starts `sandbox-agent server` in the sandbox.
5. Connects with `SandboxAgent.connect(...)` and creates a session.

## Why use Sandbox Agent SDK?

- Session lifecycle helpers (`createSession`, `resumeSession`, `destroySession`)
- Prompt/send helpers (`session.prompt`, `session.send`)
- Event streaming (`session.onEvent`) without writing transport plumbing

## Setup

1. Install dependencies:

```bash
npm install
```

2. Create a `.env` file:

```bash
E2B_API_KEY=your_e2b_api_key

# Provide at least one provider key, unless SANDBOX_AGENT points to a compatible preinstalled agent
OPENAI_API_KEY=your_openai_api_key
# CODEX_API_KEY=your_codex_api_key
# ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: any specific agent id
# SANDBOX_AGENT=codex
```

3. Run the example:

```bash
npm run start
```

The script prints an Inspector URL once the server and session are ready, then exits after cleanup.

## Notes

- If `SANDBOX_AGENT` is set, that exact agent ID is used.
- If `SANDBOX_AGENT` is not set, the script defaults to `codex` when OpenAI/Codex keys exist, otherwise `claude` when Anthropic key exists.
- API keys are available to processes running inside the sandbox.
