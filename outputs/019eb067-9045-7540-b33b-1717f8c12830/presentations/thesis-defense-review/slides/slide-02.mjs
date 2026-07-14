import { C, addBackground, addFooter, addHeader, addLabel, addPanel } from "./theme.mjs";

function branch(slide, ctx, x, title, formula, detail, accent) {
  addPanel(slide, ctx, { left: x, top: 202, width: 430, height: 208 }, {
    fill: C.panel,
    line: accent,
    lineWidth: 2,
  });
  addLabel(slide, ctx, title, { left: x + 24, top: 219, width: 380, height: 36 }, {
    fontSize: 22,
    color: accent,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(slide, ctx, formula, { left: x + 24, top: 270, width: 380, height: 54 }, {
    fontSize: 24,
    color: C.white,
    bold: true,
    mono: true,
    align: "center",
  });
  addLabel(slide, ctx, detail, { left: x + 24, top: 335, width: 380, height: 56 }, {
    fontSize: 16,
    color: C.text,
    align: "center",
    valign: "top",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  addBackground(slide, ctx);
  addHeader(
    slide,
    ctx,
    "Q2 | MISSING ACTIONS",
    "The base token separates 'action unavailable' from a valid zero control.",
    "02 / 03",
  );

  addLabel(
    slide,
    ctx,
    "One shared action-token position; availability changes only the added projected component.",
    { left: 56, top: 143, width: 1168, height: 38 },
    {
      fontSize: 19,
      color: C.text,
      align: "center",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  branch(
    slide,
    ctx,
    74,
    "Action unavailable",
    "token = b_action",
    "Passive video, first frame,\nindependent frame, or action dropout",
    C.red,
  );
  branch(
    slide,
    ctx,
    558,
    "Action available",
    "token = b_action + W a + c",
    "Normalized 7D UR5 transition\nfrom frame t-1 to frame t",
    C.green,
  );

  addPanel(slide, ctx, { left: 1022, top: 202, width: 202, height: 208 }, {
    fill: C.panel2,
    line: C.amber,
    lineWidth: 1,
  });
  addLabel(slide, ctx, "Why not a sentinel?", { left: 1040, top: 219, width: 166, height: 34 }, {
    fontSize: 18,
    color: C.amber,
    bold: true,
    align: "center",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(
    slide,
    ctx,
    "0 is a valid / mean control\n\n100 is out of distribution\n\nBoth depend on arbitrary scaling",
    { left: 1040, top: 264, width: 166, height: 128 },
    {
      fontSize: 15,
      color: C.text,
      align: "center",
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  addPanel(slide, ctx, { left: 74, top: 451, width: 1150, height: 150 }, {
    fill: C.panel2,
    line: C.blue,
    lineWidth: 1,
  });
  addLabel(slide, ctx, "What passive video teaches", { left: 98, top: 469, width: 280, height: 32 }, {
    fontSize: 19,
    color: C.blue,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(
    slide,
    ctx,
    "It does not teach 'ignore motion'. It teaches visual dynamics conditioned on history with no observed robot command.",
    { left: 98, top: 505, width: 1075, height: 34 },
    {
      fontSize: 20,
      color: C.white,
      bold: true,
      align: "center",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );
  addLabel(
    slide,
    ctx,
    "A zero-filled action plus an explicit availability bit would also be valid; this thesis did not ablate that alternative.",
    { left: 98, top: 551, width: 1075, height: 30 },
    {
      fontSize: 15,
      color: C.muted,
      align: "center",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  addFooter(slide, ctx, "Sources: thesis Sec. 3.4 and 3.8; backbone.py action-token construction");
  return slide;
}
