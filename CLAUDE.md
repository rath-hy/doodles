# doodles

A playground for aimless, curiosity-driven creation. The point is the habit of making things, not the output.

- Each project lives in its own top-level folder named `YYYY-MM-DD-short-name` (start date).
- Projects are self-contained: no shared code or dependencies between folders.
- Favor getting something working and playful quickly over polish or completeness.
- When starting a project, add a brief README in its folder noting what sparked it.

## Publishing to puthyrathy.com/doodles

Whenever a doodle can run as a static page, publish it to the personal site at `~/Desktop/puthyrathy_dot_com/puthyrathy-site` (plain HTML, no build step).

1. Copy the doodle's static output to `puthyrathy-site/doodles/<short-name>/` (drop the date prefix from the URL). If the doodle has a build step, build it first and copy only the output, with asset paths relative.
2. Every doodle page is full screen with no site header. Add `<script src="../menu.js"></script>` before `</body>` to get the floating ☰ menu back to the rest of the site.
3. Add an entry to the `doodles` array in `puthyrathy-site/doodles/index.html`: `{ slug, title, date, blurb }`.
4. Don't commit or push the site repo unless asked.

## Session log

Friends follow progress and consistency on puthyrathy.com/doodles. At the end of every session, add an entry to the `log` array in `puthyrathy-site/doodles/index.html`: `{ date: 'YYYY-MM-DD', text }`, where `text` is one plain paragraph in Rath's voice (first person, minimal formatting) summarizing what happened. Each logged date lights up a square in the page's calendar grid.
