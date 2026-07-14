export const C = {
  bg: "#0B1020",
  panel: "#141B2D",
  panel2: "#1A2338",
  white: "#F7F9FC",
  text: "#D8DFEA",
  muted: "#8F9BB3",
  line: "#46516A",
  blue: "#4FA3FF",
  blueDark: "#183E68",
  cyan: "#59D5E0",
  purple: "#A985FF",
  red: "#FF6B6B",
  amber: "#FFB454",
  green: "#69D59A",
};

export function addBackground(slide, ctx) {
  return ctx.addShape(slide, {
    left: 0,
    top: 0,
    width: ctx.W,
    height: ctx.H,
    fill: { type: "solid", color: C.bg },
    line: ctx.line(C.bg, 0),
    name: "background",
  });
}

export function addHeader(slide, ctx, kicker, title, page) {
  ctx.addShape(slide, {
    left: 56,
    top: 43,
    width: 26,
    height: 4,
    fill: { type: "solid", color: C.blue },
    line: ctx.line(C.blue, 0),
    name: "kicker-marker",
  });
  ctx.addText(slide, {
    text: kicker,
    left: 92,
    top: 28,
    width: 360,
    height: 34,
    fontSize: 14,
    color: C.blue,
    bold: true,
    valign: "middle",
    name: "kicker-label",
  });
  ctx.addText(slide, {
    text: title,
    left: 56,
    top: 68,
    width: 1160,
    height: 64,
    fontSize: 40,
    color: C.white,
    bold: true,
    typeface: ctx.fonts.title,
    valign: "middle",
    name: "claim-title",
  });
  ctx.addText(slide, {
    text: page,
    left: 1170,
    top: 674,
    width: 54,
    height: 24,
    fontSize: 12,
    color: C.muted,
    align: "right",
    valign: "middle",
    name: "page-marker",
  });
}

export function addPanel(slide, ctx, frame, options = {}) {
  return ctx.addShape(slide, {
    ...frame,
    geometry: options.geometry || "roundRect",
    fill: { type: "solid", color: options.fill || C.panel },
    line: ctx.line(options.line || C.line, options.lineWidth ?? 1),
    name: options.name,
  });
}

export function addLabel(slide, ctx, text, frame, options = {}) {
  return ctx.addText(slide, {
    text,
    ...frame,
    fontSize: options.fontSize || 18,
    color: options.color || C.text,
    bold: options.bold || false,
    typeface: options.mono ? ctx.fonts.mono : (options.typeface || ctx.fonts.body),
    align: options.align || "left",
    valign: options.valign || "middle",
    insets: options.insets || { left: 10, right: 10, top: 8, bottom: 8 },
    name: options.name,
  });
}

export function connect(slide, from, to, color = C.line) {
  const connector = slide.shapes.connect(from, to, {
    kind: "straight",
    fromSide: "right",
    toSide: "left",
    line: { style: "solid", fill: color, width: 2 },
    tail: { type: "arrow", width: "med", length: "med" },
  });
  connector.bringToFront();
  return connector;
}

export function addFooter(slide, ctx, text) {
  ctx.addText(slide, {
    text,
    left: 56,
    top: 672,
    width: 980,
    height: 24,
    fontSize: 11,
    color: C.muted,
    valign: "middle",
    name: "source-footer",
  });
}
