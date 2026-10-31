# Writing style

These rules apply to the README, everything in `docs/`, code comments and commit messages.
They follow how mature systems projects (runc, crun, gVisor, Firecracker) write, and they
avoid the habits that make text read as machine-generated.

## Do

- Start a page with one plain sentence that says what the thing is.
- Prefer short sentences and common words. Write "is" and "has".
- Give numbers with their setup (hardware, driver, what was measured) and link the raw data.
- Say what the project does not do, and what is untested.
- Use sentence case for headings: "Building from source", not "Building From Source".
- Use code formatting for commands, paths, env vars, function names.
- Use a table when you compare things. Use a list when order or count matters.
- Write "we" when the project makes a choice, "you" when the reader acts.

## Avoid

- Em dashes. Use a comma, a colon, parentheses, or a new sentence.
- Bold inside sentences. Bold is for a rare warning, at most once per page.
- Lists where every item starts with a bold label followed by a colon.
- Groups of three for rhythm ("fast, simple, and secure").
- "Not X, but Y" constructions and "it's not just X".
- Endings like "..., ensuring reliability" or "..., making it easy to ...".
- Filler and hype words: seamless, robust, powerful, leverage, crucial, pivotal, delve, unlock,
  empower, effortless, cutting-edge, game-changing, best-in-class, blazing, comprehensive.
- Emoji and decorative symbols in tables (write "yes", "no", "partial").
- Headings like "Why it matters", "Key features", "Conclusion".
- Restating the same point at the start and end of a section.

## Check

`tools/style-check.sh` greps for the patterns above. It is a hint, not a gate.
