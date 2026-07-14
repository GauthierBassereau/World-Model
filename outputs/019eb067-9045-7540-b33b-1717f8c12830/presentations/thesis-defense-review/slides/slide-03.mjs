import { C, addBackground, addFooter, addHeader, addLabel, addPanel } from "./theme.mjs";

function fixBox(slide, ctx, x, title, detail, accent) {
  addPanel(slide, ctx, { left: x, top: 532, width: 276, height: 104 }, {
    fill: C.panel2,
    line: accent,
    lineWidth: 1,
  });
  addLabel(slide, ctx, title, { left: x + 14, top: 544, width: 248, height: 28 }, {
    fontSize: 16,
    color: accent,
    bold: true,
    align: "center",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(slide, ctx, detail, { left: x + 14, top: 574, width: 248, height: 50 }, {
    fontSize: 13,
    color: C.text,
    align: "center",
    valign: "top",
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
}

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  addBackground(slide, ctx);
  addHeader(
    slide,
    ctx,
    "Q3 | CEM TRAJECTORY",
    "The hook is allowed by the cost: CEM is endpoint-seeking, not path-following.",
    "03 / 03",
  );

  addPanel(slide, ctx, { left: 56, top: 150, width: 610, height: 336 }, {
    fill: C.panel,
    line: C.blue,
    lineWidth: 1,
  });
  addLabel(slide, ctx, "Implemented objective", { left: 80, top: 168, width: 260, height: 32 }, {
    fontSize: 20,
    color: C.blue,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(
    slide,
    ctx,
    "J = terminal DINO MSE\n    + 0.2 mean ||a_t||^2",
    { left: 80, top: 213, width: 562, height: 76 },
    {
      fontSize: 27,
      color: C.white,
      bold: true,
      mono: true,
      align: "center",
    },
  );
  addLabel(slide, ctx, "Optimized", { left: 84, top: 312, width: 125, height: 28 }, {
    fontSize: 17,
    color: C.green,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(slide, ctx, "final visual state; small action magnitude", { left: 228, top: 305, width: 410, height: 38 }, {
    fontSize: 17,
    color: C.text,
  });
  addLabel(slide, ctx, "Not optimized", { left: 84, top: 355, width: 125, height: 28 }, {
    fontSize: 17,
    color: C.red,
    bold: true,
    insets: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  addLabel(
    slide,
    ctx,
    "reference path; action differences; curvature; acceleration;\njerk; IK; joint limits; collisions; model uncertainty",
    { left: 228, top: 347, width: 410, height: 58 },
    {
      fontSize: 15,
      color: C.text,
      valign: "top",
    },
  );
  addPanel(slide, ctx, { left: 80, top: 420, width: 562, height: 48 }, {
    fill: C.amber,
    line: C.amber,
    lineWidth: 0,
  });
  addLabel(
    slide,
    ctx,
    "Goal: 4 s later | Plan: 10 steps at 5 Hz = 2 s",
    { left: 88, top: 424, width: 546, height: 40 },
    {
      fontSize: 19,
      color: C.bg,
      bold: true,
      align: "center",
      insets: { left: 0, right: 0, top: 0, bottom: 0 },
    },
  );

  addPanel(slide, ctx, { left: 694, top: 150, width: 530, height: 336 }, {
    fill: "#FFFFFF",
    line: C.line,
    lineWidth: 1,
  });
  await ctx.addImage(slide, {
    path: `${ctx.workspaceDir}/assets/cem_iterations_24_29.png`,
    alt: "CEM trajectory plots at iterations 24 and 29 showing a hooked planned path",
    left: 704,
    top: 160,
    width: 510,
    height: 316,
    fit: "contain",
    name: "cem-evidence",
  });

  fixBox(slide, ctx, 56, "Match the horizon", "Use 20 actions for a 4 s reference at 5 Hz.", C.green);
  fixBox(slide, ctx, 352, "Add path smoothness", "Intermediate cost plus delta-action, acceleration, and jerk penalties.", C.purple);
  fixBox(slide, ctx, 648, "Enforce feasibility", "Joint-space / IK planning, limits, workspace, and collision checks.", C.red);
  fixBox(slide, ctx, 944, "Close the loop", "Receding-horizon MPC and uncertainty / OOD penalties.", C.blue);

  addFooter(slide, ctx, "Sources: thesis Fig. 4.6; planning objective; horizon and 5 Hz configuration");
  return slide;
}
