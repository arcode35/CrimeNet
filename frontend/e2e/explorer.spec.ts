import { expect, test } from "@playwright/test";

test("primary explorer and model flow", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const runtimeErrors: string[] = [];
  page.on("pageerror", (error) => runtimeErrors.push(error.message));
  page.on("console", (message) => {
    const source = message.location().url;
    if (message.type() === "error" && !source.endsWith("/favicon.ico")) {
      runtimeErrors.push(`${source || "browser"}: ${message.text()}`);
    }
  });

  await page.goto("/explorer");
  await expect(page.getByText("DEVELOPMENT FIXTURE")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-map-loaded", "true", {
    timeout: 15_000,
  });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready", {
    timeout: 15_000,
  });
  await expect(page.locator("canvas").first()).toBeVisible({ timeout: 15_000 });
  await expect(
    page.locator(".map-legend").getByText("PREDICTED INTENSITY", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("explorer.png"), fullPage: true });

  const timeBeforeBasemapSwitch = new URL(page.url()).searchParams.get("time");
  await page.getByRole("button", { name: "SATELLITE" }).click();
  await expect(page.getByRole("button", { name: "SATELLITE" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-basemap-mode", "satellite");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-satellite-layer-visible", "true");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-satellite-loaded", "true", {
    timeout: 15_000,
  });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready");
  expect(new URL(page.url()).searchParams.get("time")).toBe(timeBeforeBasemapSwitch);
  await expect(page.locator(".maplibregl-ctrl-attrib")).toBeVisible();
  await expect(page.locator(".maplibregl-ctrl-attrib")).toContainText("MapTiler");
  await page.screenshot({ path: testInfo.outputPath("explorer-satellite.png"), fullPage: true });

  const viewport = page.viewportSize();
  if (viewport && testInfo.project.name === "desktop-1440") {
    await page.mouse.click(viewport.width * 0.55, viewport.height * 0.57);
    await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
    await expect(page.getByText("EVENT INTENSITY")).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("satellite-selected.png"), fullPage: true });
  }
  await page.getByRole("button", { name: "3D" }).click();
  await expect(page.getByRole("button", { name: "3D" })).toHaveClass(/active/);
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready");
  await page.getByRole("button", { name: "DARK" }).click();
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-basemap-mode", "dark");
  if (testInfo.project.name === "desktop-1440") {
    await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
    await page.keyboard.press("Escape");
  }

  if (viewport) {
    await page.mouse.click(viewport.width * 0.55, viewport.height * 0.57);
    await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
    await expect(page.getByText("CRIME MIX")).toBeVisible();
    await expect(page.getByText("FIXTURE DATA")).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("inspector.png"), fullPage: true });
    if (testInfo.project.name === "desktop-1440") {
      const timeBefore = new URL(page.url()).searchParams.get("time");
      const comparisonBar = page.locator(".crime-mix .distribution-track i").nth(1);
      const barWidthBefore = await comparisonBar.getAttribute("style");
      await page.keyboard.press("ArrowRight");
      await expect.poll(() => new URL(page.url()).searchParams.get("time")).not.toBe(timeBefore);
      await expect.poll(() => comparisonBar.getAttribute("style")).not.toBe(barWidthBefore);
      await page.getByRole("button", { name: "INTENSITY", exact: true }).click();
      await expect(page.getByText("λfamily · events / cell / hour")).toBeVisible();
      await page.getByRole("button", { name: "VIEW ALL 87 →" }).click();
      await expect(page.getByRole("dialog")).toBeVisible();
      await page.getByPlaceholder("Search modeled crime type...").fill("theft from vehicle");
      await expect(page.getByRole("dialog").getByText("Theft From Vehicle")).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath("all-87-types.png"), fullPage: true });
      await page.getByRole("button", { name: "Close all crime types" }).click();
    }
    await page.keyboard.press("Escape");
  }

  await page.goto("/model");
  await expect(page.getByText("SUPPORTED GEOGRAPHIES")).toBeVisible();
  await expect(page.getByText("Evaluation artifact API required")).toBeVisible();
  expect(runtimeErrors).toEqual([]);
});

test("satellite tile failure returns safely to dark", async ({ page }, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-1440",
    "One deterministic recovery check is sufficient",
  );
  await page.route("https://api.maptiler.com/maps/satellite-v4/**", (route) => {
    if (route.request().url().includes("tiles.json")) return route.continue();
    return route.abort();
  });
  await page.goto("/explorer");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-map-loaded", "true", {
    timeout: 15_000,
  });
  await page.getByRole("button", { name: "SATELLITE" }).click();
  await expect(
    page.getByText("Satellite imagery unavailable. Using standard basemap."),
  ).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByRole("button", { name: "DARK" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready");
});

test("prediction surface survives a basemap style failure", async ({ page }, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-1440",
    "One deterministic recovery check is sufficient",
  );
  await page.route("https://api.maptiler.com/**", (route) => route.abort());
  await page.goto("/explorer");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-map-fallback", "true", {
    timeout: 15_000,
  });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready", {
    timeout: 15_000,
  });
  const viewport = page.viewportSize();
  if (viewport) await page.mouse.click(viewport.width * 0.55, viewport.height * 0.57);
  await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
});
