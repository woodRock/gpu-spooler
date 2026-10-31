// Terminal output as styled lines: SGR colours and bold/dim are kept, other escape sequences are
// dropped, and a carriage return (progress bars) keeps only what was drawn last on the line.

export type Seg = { text: string; fg?: number; bold?: boolean; dim?: boolean };
export type Line = Seg[];

// CSI ... final byte, or OSC ... BEL / ESC \
const ESCAPE = /\x1b\[([0-9;?]*)([@-~])|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]/g;

type Style = { fg?: number; bold?: boolean; dim?: boolean };

function applySgr(style: Style, params: string): Style {
  const codes = params === "" ? [0] : params.split(";").map((p) => Number(p) || 0);
  let s = { ...style };
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i];
    if (c === 0) s = {};
    else if (c === 1) s.bold = true;
    else if (c === 2) s.dim = true;
    else if (c === 22) s.bold = s.dim = false;
    else if ((c >= 30 && c <= 37) || (c >= 90 && c <= 97)) s.fg = c;
    else if (c === 39) s.fg = undefined;
    else if (c === 38 || c === 48) i += codes[i + 1] === 5 ? 2 : codes[i + 1] === 2 ? 4 : 0; // 256/RGB: skipped
  }
  return s;
}

/** Parse `text` into lines. `style` is the style the text starts in; the style it ends in is
 * returned, so appended text can continue from it. */
export function parseAnsi(text: string, style: Style = {}): { lines: Line[]; style: Style } {
  const lines: Line[] = [];
  for (let raw of text.split("\n")) {
    if (raw.endsWith("\r")) raw = raw.slice(0, -1);
    const cr = raw.lastIndexOf("\r");
    if (cr >= 0) raw = raw.slice(cr + 1);
    const line: Line = [];
    let last = 0;
    const push = (t: string) => t && line.push({ text: t, ...style });
    for (const m of raw.matchAll(ESCAPE)) {
      push(raw.slice(last, m.index));
      if (m[2] === "m") style = applySgr(style, m[1] ?? "");
      last = m.index! + m[0].length;
    }
    push(raw.slice(last));
    lines.push(line);
  }
  return { lines, style };
}

export function lineText(line: Line): string {
  return line.map((s) => s.text).join("");
}
