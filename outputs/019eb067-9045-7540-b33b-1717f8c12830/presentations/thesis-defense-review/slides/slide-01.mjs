import { C, addBackground, addFooter, addHeader, addLabel, addPanel, connect } from "./theme.mjs";

function stage(slide, ctx, x, width, title, body, accent, name) {
  const panel = addPanel(slide, ctx, { left: x, top: 156, width, height: 120 }, {
    fill: C.panel,
    line: accent,
    lineWidth: 2,
    name,
  });
  ctx.addShape(slide, {
    left: x,
    top: 156,
    width: 8,
    height: 120,
    fill: { type: "solid", color: accent },
    line: ctx.line(accent, 0),
  });
  addLabel(slide, ctx, title, { left: x + 18, top: 169, width: width - 30, height: 34 }, {
    fontSize: 19,
    color: C.white,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(slide, ctx, body, { left: x + 18, top: 207, width: width - 30, height: 55 }, {
    fontSize: 16,
    color: C.text,
    mono: true,
    valign: "top",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  return panel;
}

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  addBackground(slide, ctx);
  addHeader(
    slide,
    ctx,
    "Q1 | BEFORE ATTENTION",
    "A 32-frame sample becomes B x 32 x 262 x 1024, then attention is factorized.",
    "01 / 03",
  );

  const s1 = stage(slide, ctx, 56, 184, "32 RGB frames", "5 Hz\n224 x 224", C.green, "stage-rgb");
  const s2 = stage(slide, ctx, 278, 188, "Frozen DINOv2", "32 x 256 x 768\nCLS + DINO regs removed", C.blue, "stage-dino");
  const s3 = stage(slide, ctx, 504, 210, "Per-frame corruption", "x_t = s_t z_t + (1-s_t)e\none s_t per frame", C.purple, "stage-noise");
  const s4 = stage(slide, ctx, 752, 220, "Build frame tokens", "6 prefix + 256 patches\nall width 1024", C.red, "stage-tokenize");
  const s5 = stage(slide, ctx, 1010, 214, "Pre-attention tensor", "B x 32 x 262 x 1024\n8,384 tokens / sample", C.amber, "stage-output");
  connect(slide, s1, s2);
  connect(slide, s2, s3);
  connect(slide, s3, s4);
  connect(slide, s4, s5);

  addLabel(slide, ctx, "Per-frame token accounting", { left: 56, top: 310, width: 340, height: 34 }, {
    fontSize: 18,
    color: C.white,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });

  const stripY = 355;
  const stripH = 78;
  const segments = [
    { x: 56, w: 92, color: C.purple, top: "1", bottom: "signal" },
    { x: 150, w: 104, color: C.red, top: "1", bottom: "action" },
    { x: 256, w: 174, color: C.cyan, top: "4", bottom: "registers" },
    { x: 432, w: 540, color: C.blue, top: "256", bottom: "DINO patches" },
  ];
  for (const seg of segments) {
    addPanel(slide, ctx, { left: seg.x, top: stripY, width: seg.w, height: stripH }, {
      fill: seg.color,
      line: seg.color,
      lineWidth: 1,
    });
    addLabel(slide, ctx, seg.top, { left: seg.x, top: stripY + 4, width: seg.w, height: 30 }, {
      fontSize: 24,
      color: C.bg,
      bold: true,
      mono: true,
      align: "center",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
    addLabel(slide, ctx, seg.bottom, { left: seg.x, top: stripY + 41, width: seg.w, height: 24 }, {
      fontSize: 14,
      color: C.bg,
      bold: true,
      align: "center",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    });
  }
  addLabel(slide, ctx, "= 262 tokens / frame", { left: 985, top: 364, width: 235, height: 60 }, {
    fontSize: 23,
    color: C.white,
    bold: true,
    mono: true,
    align: "center",
  });

  addPanel(slide, ctx, { left: 56, top: 470, width: 550, height: 148 }, {
    fill: C.panel2,
    line: C.blue,
    lineWidth: 1,
  });
  addLabel(slide, ctx, "Factorized attention reshapes", { left: 78, top: 485, width: 310, height: 30 }, {
    fontSize: 18,
    color: C.white,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(slide, ctx, "Spatial layers", { left: 78, top: 528, width: 150, height: 28 }, {
    fontSize: 16,
    color: C.blue,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(slide, ctx, "(B x 32, 262, 1024)", { left: 248, top: 522, width: 320, height: 38 }, {
    fontSize: 22,
    color: C.white,
    mono: true,
  });
  addLabel(slide, ctx, "Temporal layers", { left: 78, top: 570, width: 150, height: 28 }, {
    fontSize: 16,
    color: C.purple,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(slide, ctx, "(B x 262, 32, 1024)", { left: 248, top: 564, width: 320, height: 38 }, {
    fontSize: 22,
    color: C.white,
    mono: true,
  });

  addPanel(slide, ctx, { left: 634, top: 470, width: 590, height: 148 }, {
    fill: C.panel2,
    line: C.amber,
    lineWidth: 1,
  });
  addLabel(slide, ctx, "Noise and masks", { left: 656, top: 485, width: 220, height: 30 }, {
    fontSize: 18,
    color: C.white,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(
    slide,
    ctx,
    "- Gaussian latent noise; no patch masking\n- causal + 24-frame temporal window\n- 30% independent-frame temporal block\n- action mask; padding excluded from loss",
    { left: 656, top: 519, width: 540, height: 88 },
    {
      fontSize: 16,
      color: C.text,
      valign: "top",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  addFooter(slide, ctx, "Sources: thesis Fig. 3.2; backbone.py; world_trainer.py; lerobot_dataset.py");
  return slide;
}
