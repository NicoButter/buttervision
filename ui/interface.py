"""
Interfaz ButterVision – Landing UI.
Navbar fija, secciones definidas y mini footer.
"""
import json
import random
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL.PngImagePlugin import PngInfo

import config
from core.advanced_pipeline import ButterVisionPipeline
from core.model_manager import ModelManager


UI_CSS = """
/* ═══════════════════════════════════════════════════════════
   ButterVision – Landing UI
═══════════════════════════════════════════════════════════ */

html { scroll-behavior: smooth; }
*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container {
    background:
        radial-gradient(circle at 14%  6%, rgba(0, 229, 255, .16) 0%, transparent 30%),
        radial-gradient(circle at 86% 16%, rgba(255, 43, 214, .14) 0%, transparent 30%),
        linear-gradient(135deg, #070812 0%, #0b1020 48%, #12081b 100%) !important;
    color: #e5f7ff !important;
    min-height: 100vh;
}
.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}
/* Strip padding from Gradio's inner wrappers so content fills full width */
.gradio-container .main,
.gradio-container .main > .wrap {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
    gap: 0 !important;
}
/* Rows and columns inside main: full width, no extra lateral padding */
.gradio-container .main .gap,
.gradio-container .main > .wrap > .gap {
    max-width: 100% !important;
    width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
/* Content rows (generate / output sections): add lateral padding here */
.gradio-container .main > .wrap > .gap > .block,
.gradio-container .main > .wrap > .gap > div:not([class*="bv-navbar"]):not([class*="bv-footer"]) {
    padding-left: 32px !important;
    padding-right: 32px !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}
/* Hide Gradio's built-in footer ("Built with Gradio") */
.gradio-container footer,
.gradio-container > .footer,
.built-with,
footer[class*="svelte"],
.svelte-1rjryqp:not(.bv-footer) {
    display: none !important;
}
/* grid overlay */
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,.45), transparent 65%);
}

/* ── NAVBAR ─────────────────────────────────────────────────── */
.bv-navbar {
    position: relative !important;
    z-index: 10 !important;
    width: 100vw !important;
    max-width: 100vw !important;
    /* Break out of any residual container padding */
    margin-left: calc(50% - 50vw) !important;
    margin-right: calc(50% - 50vw) !important;
    min-height: 64px !important;
    gap: 0 !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
    padding: 0 32px !important;
    background: rgba(5, 8, 20, .96) !important;
    backdrop-filter: blur(22px) !important;
    -webkit-backdrop-filter: blur(22px) !important;
    border-bottom: 1px solid rgba(0, 229, 255, .18) !important;
    box-shadow:
        0 4px 32px rgba(0, 0, 0, .44),
        inset 0 -1px 0 rgba(0, 229, 255, .06) !important;
    margin-bottom: 0 !important;
}
.bv-navbar.bv-scrolled {
    background: rgba(3, 5, 14, .96) !important;
    border-color: rgba(255, 43, 214, .22) !important;
    box-shadow:
        0 4px 40px rgba(0, 0, 0, .62),
        inset 0 -1px 0 rgba(255, 43, 214, .10) !important;
}
/* strip Gradio's default block chrome inside navbar */
.bv-navbar > div,
.bv-navbar .block,
.bv-navbar .form,
.bv-navbar .gap {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Brand */
.bv-brand {
    font-size: 22px;
    font-weight: 900;
    color: #f0f8ff;
    letter-spacing: -.5px;
    text-shadow: 0 0 22px rgba(0, 229, 255, .44);
    white-space: nowrap;
    line-height: 1;
    user-select: none;
}
.brand-accent {
    color: #00e5ff;
}

/* Nav links */
.bv-nav {
    display: flex;
    gap: 2px;
    align-items: center;
    justify-content: center;
}
.bv-nav-link {
    display: inline-flex;
    align-items: center;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: #6888a8;
    text-decoration: none;
    transition: color 140ms, background 140ms;
    cursor: pointer;
    letter-spacing: .2px;
}
.bv-nav-link:hover {
    color: #d4eeff;
    background: rgba(0, 229, 255, .09);
}
.bv-tool-dropdown {
    position: relative !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.bv-tool-dropdown::before {
    content: "TOOLS";
    position: absolute;
    left: 13px;
    top: -7px;
    z-index: 2;
    padding: 0 6px;
    border-radius: 999px;
    background: rgba(5, 8, 20, .98);
    color: rgba(0, 229, 255, .82);
    font-size: 8.5px;
    font-weight: 900;
    letter-spacing: 1.4px;
    line-height: 1;
    pointer-events: none;
}
.bv-navbar .bv-tool-dropdown .wrap {
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 14px !important;
    border-radius: 12px !important;
    background:
        linear-gradient(135deg, rgba(0, 229, 255, .13), rgba(255, 43, 214, .08)),
        rgba(4, 8, 20, .84) !important;
    border: 1px solid rgba(0, 229, 255, .32) !important;
    box-shadow:
        0 0 0 1px rgba(255, 255, 255, .035) inset,
        0 10px 28px rgba(0, 0, 0, .22),
        0 0 24px rgba(0, 229, 255, .08) !important;
    color: #d9f6ff !important;
    transition: border-color 150ms, box-shadow 150ms, background 150ms !important;
}
.bv-navbar .bv-tool-dropdown .wrap:hover,
.bv-navbar .bv-tool-dropdown .wrap:focus-within {
    border-color: rgba(255, 43, 214, .48) !important;
    background:
        linear-gradient(135deg, rgba(0, 229, 255, .18), rgba(255, 43, 214, .12)),
        rgba(4, 8, 20, .92) !important;
    box-shadow:
        0 0 0 1px rgba(255, 255, 255, .05) inset,
        0 12px 34px rgba(0, 0, 0, .30),
        0 0 30px rgba(255, 43, 214, .12) !important;
}
.bv-navbar .bv-tool-dropdown input,
.bv-navbar .bv-tool-dropdown .single-select,
.bv-navbar .bv-tool-dropdown [data-testid="dropdown"] {
    color: #ecfbff !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    letter-spacing: .2px !important;
}

/* Navbar model controls */
.bv-navbar label { display: none !important; }
.bv-navbar .wrap {
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 10px !important;
    font-size: 13px !important;
    background: rgba(255, 255, 255, .055) !important;
    border: 1px solid rgba(0, 229, 255, .22) !important;
    color: #c0daf0 !important;
    border-radius: 8px !important;
    margin: 0 !important;
}
.bv-navbar .wrap:focus-within {
    border-color: rgba(0, 229, 255, .55) !important;
    box-shadow: 0 0 0 2px rgba(0, 229, 255, .12) !important;
}

/* Refresh button */
.bv-refresh-btn button {
    height: 36px !important;
    width: 36px !important;
    min-width: 36px !important;
    max-width: 36px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0, 229, 255, .26) !important;
    background: rgba(0, 229, 255, .06) !important;
    color: #60b8d8 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    transition: background 130ms, border-color 130ms, color 130ms;
}
.bv-refresh-btn button:hover {
    background: rgba(0, 229, 255, .18) !important;
    border-color: rgba(0, 229, 255, .52) !important;
    color: #c8f2ff !important;
}

/* Model status pill */
.model-status {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    white-space: nowrap;
    border: 1px solid;
    line-height: 1;
}
.model-status-ok   { color: #96ffd1; background: rgba(0, 255, 170, .09);  border-color: rgba(0, 255, 170, .36); }
.model-status-bad  { color: #ffb4c8; background: rgba(255, 43, 84, .10);  border-color: rgba(255, 43, 84, .38); }
.model-status-warn { color: #ffe7a3; background: rgba(255, 198, 41, .09); border-color: rgba(255, 198, 41, .32); }

/* ── SECTION SEPARATORS ──────────────────────────────────────── */
.bv-section-sep {
    padding: 52px 32px 24px;
    scroll-margin-top: 72px;
}
.bv-section-eyebrow {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 2.8px;
    text-transform: uppercase;
    color: #00e5ff;
    opacity: .75;
    margin-bottom: 7px;
}
.bv-section-heading {
    font-size: 22px;
    font-weight: 800;
    color: #eef6ff;
    margin-bottom: 18px;
    line-height: 1.2;
    letter-spacing: -.3px;
}
.bv-section-rule {
    height: 1px;
    background: linear-gradient(
        90deg,
        rgba(0, 229, 255, .38) 0%,
        rgba(255, 43, 214, .22) 42%,
        transparent 72%
    );
}

/* ── CARDS ───────────────────────────────────────────────────── */
.bv-card {
    border: 1px solid rgba(0, 229, 255, .14) !important;
    border-radius: 14px !important;
    background: linear-gradient(160deg, rgba(14, 22, 44, .90), rgba(8, 12, 26, .95)) !important;
    box-shadow:
        0 24px 64px rgba(0, 0, 0, .40),
        inset 0 0 0 1px rgba(255, 255, 255, .028) !important;
    padding: 22px 22px 18px !important;
    transition: border-color 200ms;
}
.bv-card:hover { border-color: rgba(0, 229, 255, .28) !important; }
.bv-card-params { border-color: rgba(255, 43, 214, .16) !important; }
.bv-card-params:hover { border-color: rgba(255, 43, 214, .32) !important; }

.bv-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13.5px;
    font-weight: 800;
    color: #cce4f8;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, .055);
    letter-spacing: .2px;
}
.bv-card-icon { font-size: 11px; color: #00e5ff; }

/* ── CARD BACKGROUNDS ──────────────────────────────────────────── */
.bv-card {
    background: rgba(9, 14, 30, .92) !important;
}
.bv-card-params {
    background: rgba(10, 10, 26, .94) !important;
}

/* ── LABEL OVERRIDES (strip Gradio's purple pill) ──────────────── */
.bv-card label,
.bv-card label > span,
.bv-card .label-wrap,
.bv-card .label-wrap > span,
.bv-card .block > label,
.bv-card .block > label > span,
.bv-card [data-testid] > label > span {
    display: inline !important;
    padding: 0 !important;
    margin: 0 0 4px 0 !important;
    background: none !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #6a8aaa !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    letter-spacing: .4px !important;
    text-transform: uppercase !important;
    box-shadow: none !important;
}

/* ── INPUTS & TEXTAREAS ────────────────────────────────────────── */
.bv-card textarea,
.bv-card input[type="text"],
.bv-card input[type="number"],
.bv-card .wrap {
    background: rgba(3, 6, 18, .80) !important;
    color: #d8eeff !important;
    border: 1px solid rgba(0, 229, 255, .16) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    transition: border-color 150ms, box-shadow 150ms !important;
}
.bv-card textarea:focus,
.bv-card input[type="text"]:focus,
.bv-card input[type="number"]:focus {
    border-color: rgba(0, 229, 255, .50) !important;
    box-shadow: 0 0 0 2px rgba(0, 229, 255, .09), 0 0 14px rgba(0, 229, 255, .07) !important;
    outline: none !important;
}
/* Number input (Seed) */
.bv-card input[type="number"] {
    -moz-appearance: textfield !important;
}

/* ── SLIDERS ───────────────────────────────────────────────────── */
.bv-card input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    height: 4px !important;
    border-radius: 4px !important;
    background: rgba(0, 229, 255, .18) !important;
    border: none !important;
    outline: none !important;
    cursor: pointer !important;
}
.bv-card input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    width: 16px !important;
    height: 16px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #00e5ff, #a040ff) !important;
    border: 2px solid rgba(255, 255, 255, .22) !important;
    box-shadow: 0 0 8px rgba(0, 229, 255, .35) !important;
    cursor: pointer !important;
    transition: transform 140ms !important;
}
.bv-card input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.18) !important;
    box-shadow: 0 0 14px rgba(0, 229, 255, .55) !important;
}
.bv-card input[type="range"]::-moz-range-thumb {
    width: 16px !important;
    height: 16px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #00e5ff, #a040ff) !important;
    border: 2px solid rgba(255, 255, 255, .22) !important;
    cursor: pointer !important;
}
/* Slider row: min/max values */
.bv-card .range-container,
.bv-card .range-min,
.bv-card .range-max {
    color: #3a5570 !important;
    font-size: 11px !important;
}
/* Slider value input box */
.bv-card .value-input input {
    background: rgba(3, 6, 18, .70) !important;
    border: 1px solid rgba(0, 229, 255, .14) !important;
    color: #9bbfd8 !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    text-align: center !important;
}

/* Generate button */
.bv-generate-btn { margin-top: 8px; }
.bv-generate-btn button {
    width: 100% !important;
    padding: 14px 0 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    letter-spacing: .4px !important;
    background: linear-gradient(90deg, #00e5ff 0%, #ff2bd6 100%) !important;
    border: none !important;
    color: #04070f !important;
    box-shadow:
        0 0 32px rgba(0, 229, 255, .22),
        0 0 64px rgba(255, 43, 214, .12) !important;
    transition: filter 140ms, box-shadow 140ms !important;
}
.bv-generate-btn button:hover {
    filter: brightness(1.10) !important;
    box-shadow:
        0 0 44px rgba(0, 229, 255, .34),
        0 0 88px rgba(255, 43, 214, .22) !important;
}

/* ── OUTPUT CARDS ────────────────────────────────────────────── */
.bv-output-card {
    border: 1px solid rgba(0, 229, 255, .14) !important;
    border-radius: 14px !important;
    background: linear-gradient(160deg, rgba(14, 22, 44, .90), rgba(8, 12, 26, .95)) !important;
    box-shadow:
        0 24px 64px rgba(0, 0, 0, .40),
        inset 0 0 0 1px rgba(255, 255, 255, .028) !important;
    padding: 22px !important;
}
.bv-output-card .image-container,
.bv-output-card img { border-radius: 10px !important; }
.bv-output-card label { color: #8aa4c0 !important; font-size: 12px !important; font-weight: 600 !important; }

.bv-info-card {
    border: 1px solid rgba(255, 43, 214, .14) !important;
    border-radius: 14px !important;
    background: linear-gradient(160deg, rgba(14, 22, 44, .90), rgba(8, 12, 26, .95)) !important;
    box-shadow:
        0 24px 64px rgba(0, 0, 0, .40),
        inset 0 0 0 1px rgba(255, 255, 255, .028) !important;
    padding: 22px !important;
}
.bv-info-card label { color: #8aa4c0 !important; font-size: 12px !important; font-weight: 600 !important; }
.bv-info-card textarea {
    background: rgba(4, 9, 22, .76) !important;
    color: #b8d0e8 !important;
    border-color: rgba(255, 43, 214, .16) !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* ── MODAL ───────────────────────────────────────────────────── */
.bv-image-modal {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 28px;
    background:
        radial-gradient(circle at 50% 42%, rgba(0, 229, 255, .16), transparent 38%),
        rgba(2, 5, 14, .80);
    backdrop-filter: blur(20px);
    opacity: 0;
    pointer-events: none;
    transition: opacity 220ms ease;
}
.bv-image-modal.bv-modal-open { opacity: 1; pointer-events: auto; }
.bv-modal-shell {
    width: min(92vw, 980px);
    max-height: 92vh;
    border: 1px solid rgba(0, 229, 255, .34);
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(10, 17, 34, .97), rgba(5, 8, 20, .99));
    box-shadow:
        0 0 64px rgba(0, 229, 255, .22),
        0 0 100px rgba(255, 43, 214, .12),
        0 32px 80px rgba(0, 0, 0, .66);
    transform: translateY(20px) scale(0.96);
    transition: transform 260ms cubic-bezier(.2,.8,.2,1);
    overflow: hidden;
}
.bv-image-modal.bv-modal-open .bv-modal-shell { transform: translateY(0) scale(1); }
.bv-modal-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(0, 229, 255, .16);
}
.bv-modal-title { color: #e8f6ff; font-size: 13px; font-weight: 800; }
.bv-modal-close {
    appearance: none;
    border: 1px solid rgba(255, 43, 214, .40);
    border-radius: 8px;
    background: rgba(255, 43, 214, .09);
    color: #ffd6f5;
    width: 34px;
    height: 30px;
    cursor: pointer;
    font-size: 16px;
    transition: background 130ms;
}
.bv-modal-close:hover { background: rgba(255, 43, 214, .22); }
.bv-modal-image-wrap { padding: 16px; }
#bv-modal-image {
    display: block;
    width: 100%;
    max-height: 76vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 0 32px rgba(0, 229, 255, .12);
}

/* ── FOOTER ──────────────────────────────────────────────────── */
.bv-footer {
    display: block !important;
    width: 100vw !important;
    max-width: 100vw !important;
    /* Break out of any residual container padding */
    margin-left: calc(50% - 50vw) !important;
    margin-right: calc(50% - 50vw) !important;
    margin-top: 60px;
    margin-bottom: 0 !important;
    padding: 28px 32px 32px;
    border-top: 1px solid rgba(0, 229, 255, .10);
    background: linear-gradient(0deg, rgba(2, 4, 14, .82), transparent);
}
.bv-footer-inner {
    max-width: 100%;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 14px;
}
.bv-footer-left { display: flex; flex-direction: column; gap: 4px; }
.bv-footer-brand { font-size: 16px; font-weight: 900; color: #6888a0; letter-spacing: -.3px; }
.bv-footer-brand span { color: #00e5ff; opacity: .65; }
.bv-footer-tagline { font-size: 11.5px; color: #364e66; font-weight: 500; }
.bv-footer-right { display: flex; align-items: center; gap: 20px; }
.bv-footer-tech { display: flex; gap: 6px; align-items: center; }
.bv-footer-badge {
    font-size: 10.5px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    color: #4e6880;
    border: 1px solid rgba(78, 104, 128, .28);
    letter-spacing: .5px;
}
.bv-footer-copy { font-size: 11px; color: #2c3e52; }

/* Kill any Gradio-added space after the last block */
.gradio-container > .main > .wrap > *:last-child,
.gradio-container > *:last-child {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
/* Ensure body/html end at footer with no extra scroll space */
body {
    overflow-x: hidden !important;
}
html {
    overflow-x: hidden !important;
}

/* ── RESPONSIVE ──────────────────────────────────────────────── */
@media (max-width: 900px) {
    .bv-navbar { padding: 0 16px !important; }
    .bv-nav { display: none !important; }
    .bv-section-sep { padding: 36px 16px 18px; }
    .bv-footer { padding: 22px 16px 28px !important; }
    .bv-footer-right { display: none; }
    .bv-image-modal { padding: 12px; }
}
"""

UI_JS = """
(function() {
  console.log("ButterVision JS Initializing...");
  
  const updateNavbar = () => {
    const navbar = document.querySelector('.bv-navbar');
    if (!navbar) return;
    navbar.classList.toggle('bv-scrolled', window.scrollY > 40);
  };

  window.closeBVModal = function() {
    const modal = document.getElementById('bv-image-modal');
    if (modal) modal.classList.remove('bv-modal-open');
  };

  window.openBVModal = function(src) {
    console.log("Opening modal for:", src);
    if (!src) return;
    const modal = document.getElementById('bv-image-modal');
    const modalImage = document.getElementById('bv-modal-image');
    if (!modal || !modalImage) {
        console.error("Modal elements not found");
        return;
    }
    modalImage.src = src;
    modal.classList.add('bv-modal-open');
  };

  // Click handler for closing
  document.addEventListener('click', (event) => {
    if (event.target?.matches?.('[data-bv-modal-close]')) window.closeBVModal();
    if (event.target?.id === 'bv-image-modal') window.closeBVModal();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') window.closeBVModal();
  });

  window.addEventListener('scroll', updateNavbar, { passive: true });
  
  // Initial check
  setTimeout(updateNavbar, 100);
})();
"""


class ButterVisionUI:
    """Interfaz principal enfocada únicamente en Text-to-Image."""

    def __init__(self):
        self.model_manager = ModelManager()
        self.model_choices = self._get_model_choices()
        active_model = self.model_manager.resolve_model_path(config.model_config.model_id) or config.model_config.model_id
        self.sd_manager = ButterVisionPipeline(
            model_id=active_model,
            enable_optimizations=True,
            enable_lcm=False,
        )

    def _get_model_choices(self):
        """Retorna modelos locales detectados y asegura que el activo esté presente."""
        model_paths = [model["path"] for model in self.model_manager.list_local_model_infos()]
        active_model = config.model_config.model_id
        resolved_active = self.model_manager.resolve_model_path(active_model) or active_model

        if resolved_active not in model_paths:
            model_paths.insert(0, resolved_active)

        return [(self._format_model_label(model_path), model_path) for model_path in model_paths]

    def _get_model_choice_values(self):
        """Retorna las rutas reales usadas como valores del selector."""
        return [choice[1] if isinstance(choice, tuple) else choice for choice in self.model_choices]

    def _format_model_label(self, model_path):
        """Muestra un nombre breve para el selector."""
        path = Path(model_path)
        return path.name

    def _model_status_html(self, model_id):
        """Genera indicador visual de compatibilidad modelo/VRAM."""
        info = self.model_manager.get_model_info(model_id)
        if info is None:
            return (
                "<div class='model-status model-status-warn'>"
                "Modelo no encontrado"
                "</div>"
            )

        if info["fits_gpu"] is True:
            return "<div class='model-status model-status-ok'>Compatible con la GPU</div>"
        elif info["fits_gpu"] is False:
            return "<div class='model-status model-status-bad'>No compatible con la GPU</div>"

        return "<div class='model-status model-status-warn'>Compatibilidad no determinada</div>"

    def refresh_models(self):
        """Actualiza lista de modelos locales y el indicador."""
        self.model_choices = self._get_model_choices()
        model_values = self._get_model_choice_values()
        current_model = self.sd_manager.model_id
        if current_model not in model_values:
            current_model = model_values[0] if model_values else current_model
        return (
            gr.update(choices=self.model_choices, value=current_model),
            self._model_status_html(current_model),
        )

    def select_model(self, model_id):
        """Cambia el modelo activo para la próxima generación."""
        if not model_id:
            return self._model_status_html(self.sd_manager.model_id)

        if model_id != self.sd_manager.model_id:
            self.sd_manager.change_model(model_id)
            config.model_config.model_id = model_id
            if hasattr(self, "instantid_manager") and self.instantid_manager is not None:
                self.instantid_manager.cleanup()
                self.instantid_manager = None

        return self._model_status_html(model_id)

    def switch_function(self, function_name):
        """Muestra solo el panel correspondiente a la función elegida."""
        is_face_reference = function_name == "Face Reference"
        message = ""

        if is_face_reference:
            try:
                self.sd_manager.cleanup()
                manager = self._get_instantid_manager()
                message = manager.ensure_assets(allow_download=True)
            except Exception as error:
                message = f"Error preparando Face Reference: {error}"
        elif hasattr(self, "instantid_manager") and self.instantid_manager is not None:
            self.instantid_manager.cleanup()

        return (
            gr.update(visible=not is_face_reference),
            gr.update(visible=is_face_reference),
            message,
        )

    def _get_instantid_manager(self):
        """Carga el backend Face Reference solo cuando se usa esa herramienta."""
        if not hasattr(self, "instantid_manager"):
            self.instantid_manager = None

        if self.instantid_manager is None:
            from core.face_reference_pipeline import SD15FaceReferencePipeline

            self.instantid_manager = SD15FaceReferencePipeline(
                base_model=self.sd_manager.model_id,
            )

        return self.instantid_manager

    def _get_generation_dir(self, created_at):
        """Crea un directorio único para una corrida txt2img."""
        outputs_dir = config.OUTPUTS_DIR
        outputs_dir.mkdir(parents=True, exist_ok=True)

        base_name = created_at.strftime("%d%m%Y-%H%M%S-generation")
        generation_dir = outputs_dir / base_name
        suffix = 2
        while generation_dir.exists():
            generation_dir = outputs_dir / f"{base_name}-{suffix}"
            suffix += 1

        generation_dir.mkdir(parents=True, exist_ok=False)
        return generation_dir

    def _png_metadata(self, metadata):
        """Prepara metadatos embebidos dentro del PNG."""
        png_info = PngInfo()
        png_info.add_text("ButterVision", "txt2img")
        png_info.add_text("Prompt", metadata["prompt"])
        png_info.add_text("Negative Prompt", metadata["negative_prompt"])
        png_info.add_text("Seed", str(metadata["seed"]))
        png_info.add_text("Model", metadata["model"])
        png_info.add_text("Generation Metadata", json.dumps(metadata, ensure_ascii=False))
        return png_info

    def _save_generation(self, images, metadata, reference_images=None):
        """Guarda PNGs, prompts y metadata de una generación."""
        created_at = datetime.fromisoformat(metadata["created_at"])
        generation_dir = self._get_generation_dir(created_at)

        metadata = {
            **metadata,
            "generation_dir": str(generation_dir),
            "images": [],
            "reference_images": [],
        }

        for name, reference_image in (reference_images or {}).items():
            reference_path = generation_dir / name
            reference_image.save(reference_path)
            metadata["reference_images"].append(str(reference_path))

        png_info = self._png_metadata(metadata)
        for index, image in enumerate(images, start=1):
            filename = f"image_{index:02d}.png"
            filepath = generation_dir / filename
            image.save(filepath, pnginfo=png_info)
            metadata["images"].append(str(filepath))

        metadata_path = generation_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        prompt_path = generation_dir / "prompt.txt"
        prompt_path.write_text(
            "\n".join(
                [
                    f"Prompt: {metadata['prompt']}",
                    f"Negative Prompt: {metadata['negative_prompt']}",
                    f"Seed: {metadata['seed']}",
                    f"Model: {metadata['model']}",
                    f"Size: {metadata['width']}x{metadata['height']}",
                    f"Steps: {metadata['steps']} | CFG: {metadata['cfg_scale']}",
                    f"Batch: {metadata['batch_size']}",
                ]
            ),
            encoding="utf-8",
        )

        return generation_dir, [Path(path) for path in metadata["images"]]

    def _load_history_gallery(self, limit=12):
        """Carga las últimas imágenes generadas desde outputs."""
        outputs_dir = config.OUTPUTS_DIR
        if not outputs_dir.exists():
            return []

        image_paths = []
        for directory in outputs_dir.iterdir():
            if directory.is_dir():
                image_paths.extend(directory.glob("*.png"))

        image_paths = sorted(image_paths, key=lambda path: path.stat().st_mtime, reverse=True)
        return [str(path) for path in image_paths[:limit]]

    def refresh_history(self):
        """Refresca la galería de generaciones recientes."""
        return self._load_history_gallery()

    def txt2img_generate(
        self,
        prompt,
        negative_prompt,
        steps,
        cfg_scale,
        width,
        height,
        seed,
        batch_size,
    ):
        """Genera una imagen desde texto."""
        try:
            prompt = (prompt or "").strip()
            negative_prompt = (negative_prompt or "").strip()

            if not prompt:
                return None, "Ingresa un prompt para generar una imagen.", seed, self._load_history_gallery()

            if seed in (None, ""):
                seed = -1

            if int(seed) == -1:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = int(seed)

            steps = int(steps)
            cfg_scale = float(cfg_scale)
            width = max(256, min(int(width), config.model_config.max_width))
            height = max(256, min(int(height), config.model_config.max_height))
            batch_size = max(1, min(int(batch_size), config.model_config.max_batch_size))

            images = self.sd_manager.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                seed=seed,
                num_images=batch_size,
            )

            created_at = datetime.now().isoformat(timespec="seconds")
            metadata = {
                "module": "txt2img",
                "created_at": created_at,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "model": self.sd_manager.model_id,
            }
            generation_dir, _ = self._save_generation(images, metadata)
            info = (
                f"Seed: {seed}\n"
                f"Size: {width}x{height}\n"
                f"Steps: {steps} | CFG: {cfg_scale:.1f}\n"
                f"Batch: {batch_size}\n"
                f"Saved: {generation_dir}"
            )
            return images[0], info, seed, self._load_history_gallery()

        except Exception as error:
            return None, f"Error: {error}", seed, self._load_history_gallery()

    def face_reference_generate(
        self,
        face_image,
        prompt,
        negative_prompt,
        steps,
        cfg_scale,
        width,
        height,
        seed,
        batch_size,
        identity_strength,
        structure_strength,
    ):
        """Genera una imagen preservando identidad facial con IP-Adapter Face."""
        try:
            if face_image is None:
                return None, "Sube una imagen de referencia con una cara clara.", seed, self._load_history_gallery()

            prompt = (prompt or "").strip()
            negative_prompt = (negative_prompt or "").strip()
            if not prompt:
                return None, "Ingresa un prompt para generar una imagen.", seed, self._load_history_gallery()

            if seed in (None, ""):
                seed = -1

            if int(seed) == -1:
                seed = random.randint(0, 2**32 - 1)
            else:
                seed = int(seed)

            steps = int(steps)
            cfg_scale = float(cfg_scale)
            width = max(512, min(int(width), config.model_config.face_max_width))
            height = max(512, min(int(height), config.model_config.face_max_height))
            batch_size = 1
            identity_strength = float(identity_strength)
            structure_strength = float(structure_strength)

            self.sd_manager.cleanup()
            manager = self._get_instantid_manager()
            images = manager.generate(
                face_image=face_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=cfg_scale,
                seed=seed,
                num_images=batch_size,
                identity_strength=identity_strength,
                structure_strength=structure_strength,
            )

            created_at = datetime.now().isoformat(timespec="seconds")
            metadata = {
                "module": "face_reference",
                "backend": "SD1.5 IP-Adapter Face",
                "created_at": created_at,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "identity_strength": identity_strength,
                "structure_strength": structure_strength,
                "model": manager.base_model,
            }
            generation_dir, _ = self._save_generation(
                images,
                metadata,
                reference_images={"reference_face.png": face_image},
            )
            info = (
                f"Mode: Face Reference / SD1.5 IP-Adapter Face\n"
                f"Seed: {seed}\n"
                f"Size: {width}x{height}\n"
                f"Steps: {steps} | CFG: {cfg_scale:.1f}\n"
                f"Identity: {identity_strength:.2f} | Structure: {structure_strength:.2f}\n"
                f"Saved: {generation_dir}"
            )
            return images[0], info, seed, self._load_history_gallery()

        except Exception as error:
            return None, f"Error: {error}", seed, self._load_history_gallery()

    def create_interface(self):
        """Crea la interfaz landing-page de ButterVision."""
        with gr.Blocks(title="ButterVision") as interface:

            active_model = self.sd_manager.model_id

            # ── NAVBAR ──────────────────────────────────────────────────
            with gr.Row(elem_classes=["bv-navbar"]):
                with gr.Column(scale=0, min_width=200):
                    gr.HTML(
                        "<div class='bv-brand'>"
                        "Butter<span class='brand-accent'>Vision</span>"
                        "</div>"
                    )
                with gr.Column(scale=1, min_width=180):
                    gr.HTML(
                        "<nav class='bv-nav'>"
                        "<a class='bv-nav-link' href='#bv-main'>Generate</a>"
                        "</nav>"
                    )
                with gr.Column(scale=2, min_width=280):
                    function_selector = gr.Dropdown(
                        choices=["Text to Image", "Face Reference"],
                        value="Text to Image",
                        label="",
                        interactive=True,
                        elem_classes=["bv-tool-dropdown"],
                    )
                with gr.Column(scale=3, min_width=340):
                    with gr.Row():
                        model_selector = gr.Dropdown(
                            choices=self.model_choices,
                            value=active_model,
                            label="",
                            scale=4,
                        )
                        refresh_models_btn = gr.Button(
                            "↻", size="sm", scale=0, min_width=36,
                            elem_classes=["bv-refresh-btn"],
                        )
                with gr.Column(scale=1, min_width=180):
                    model_status = gr.HTML(value=self._model_status_html(active_model))

            with gr.Group(visible=True) as text_to_image_panel:
                # ── SECTION: GENERATE ────────────────────────────────────
                gr.HTML(
                    "<div class='bv-section-sep' id='bv-main'>"
                    "<div class='bv-section-eyebrow'>Text to Image</div>"
                    "<div class='bv-section-heading'>Generate</div>"
                    "<div class='bv-section-rule'></div>"
                    "</div>"
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, elem_classes=["bv-card"]):
                        gr.HTML(
                            "<div class='bv-card-header'>"
                            "<span class='bv-card-icon'>✦</span> Prompt"
                            "</div>"
                        )
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image you want to generate…",
                            lines=4,
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Elements to avoid…",
                            lines=3,
                        )
                        with gr.Row(elem_classes=["bv-generate-btn"]):
                            generate_btn = gr.Button("✦  Generate", variant="primary", size="lg")

                    with gr.Column(scale=2, elem_classes=["bv-card", "bv-card-params"]):
                        gr.HTML(
                            "<div class='bv-card-header'>"
                            "<span class='bv-card-icon'>⚙</span> Parameters"
                            "</div>"
                        )
                        steps = gr.Slider(
                            minimum=1, maximum=60,
                            value=config.model_config.default_steps,
                            step=1, label="Steps",
                        )
                        cfg_scale = gr.Slider(
                            minimum=1.0, maximum=15.0,
                            value=config.model_config.default_cfg_scale,
                            step=0.5, label="CFG Scale",
                        )
                        width = gr.Slider(
                            minimum=256, maximum=config.model_config.max_width,
                            value=config.model_config.default_width,
                            step=64, label="Width",
                        )
                        height = gr.Slider(
                            minimum=256, maximum=config.model_config.max_height,
                            value=config.model_config.default_height,
                            step=64, label="Height",
                        )
                        if config.model_config.max_batch_size > 1:
                            batch_size = gr.Slider(
                                minimum=1, maximum=config.model_config.max_batch_size,
                                value=config.model_config.default_batch_size,
                                step=1, label="Batch",
                            )
                        else:
                            batch_size = gr.Number(
                                value=config.model_config.default_batch_size,
                                label="Batch",
                                precision=0,
                                interactive=False,
                            )
                        seed = gr.Number(value=-1, label="Seed  (−1 = random)", precision=0)

                # ── SECTION: OUTPUT ──────────────────────────────────────
                gr.HTML(
                    "<div class='bv-section-sep' id='bv-out'>"
                    "<div class='bv-section-eyebrow'>Result</div>"
                    "<div class='bv-section-heading'>Output</div>"
                    "<div class='bv-section-rule'></div>"
                    "</div>"
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, elem_classes=["bv-output-card"]):
                        image_output = gr.Image(
                            type="pil",
                            label="Generated Image",
                            height=512,
                            elem_id="bv-generated-image",
                        )
                    with gr.Column(scale=2, elem_classes=["bv-info-card"]):
                        info_text = gr.Textbox(
                            label="Generation Info",
                            interactive=False,
                            lines=7,
                        )

                with gr.Row(equal_height=False):
                    with gr.Column(elem_classes=["bv-output-card"]):
                        history_gallery = gr.Gallery(
                            value=self._load_history_gallery(),
                            label="Recent Generations",
                            columns=4,
                            rows=2,
                            height=360,
                            object_fit="contain",
                        )
                        refresh_history_btn = gr.Button("↻ Refresh History", size="sm")

            with gr.Group(visible=False) as face_reference_panel:
                gr.HTML(
                    "<div class='bv-section-sep' id='bv-face-ref'>"
                    "<div class='bv-section-eyebrow'>Identity Preserving</div>"
                    "<div class='bv-section-heading'>Face Reference</div>"
                    "<div class='bv-section-rule'></div>"
                    "</div>"
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, elem_classes=["bv-card"]):
                        gr.HTML(
                            "<div class='bv-card-header'>"
                            "<span class='bv-card-icon'>✦</span> Reference"
                            "</div>"
                        )
                        face_reference_image = gr.Image(
                            type="pil",
                            label="Reference Face",
                            height=360,
                        )
                        face_identity_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            value=0.8,
                            step=0.05,
                            label="Identity Strength",
                        )
                        face_structure_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            value=0.8,
                            step=0.05,
                            label="Structure Strength",
                        )

                    with gr.Column(scale=3, elem_classes=["bv-card"]):
                        gr.HTML(
                            "<div class='bv-card-header'>"
                            "<span class='bv-card-icon'>✦</span> Prompt"
                            "</div>"
                        )
                        face_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the image while IP-Adapter preserves the reference identity…",
                            lines=4,
                        )
                        face_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Elements to avoid…",
                            lines=3,
                        )
                        with gr.Row(elem_classes=["bv-generate-btn"]):
                            face_generate_btn = gr.Button("✦  Generate Face Reference", variant="primary", size="lg")

                    with gr.Column(scale=2, elem_classes=["bv-card", "bv-card-params"]):
                        gr.HTML(
                            "<div class='bv-card-header'>"
                            "<span class='bv-card-icon'>⚙</span> Parameters"
                            "</div>"
                        )
                        face_steps = gr.Slider(
                            minimum=1,
                            maximum=60,
                            value=config.model_config.face_default_steps,
                            step=1,
                            label="Steps",
                        )
                        face_cfg_scale = gr.Slider(
                            minimum=0.0,
                            maximum=15.0,
                            value=5.0,
                            step=0.5,
                            label="CFG Scale",
                        )
                        face_width = gr.Number(
                            value=config.model_config.face_default_width,
                            label="Width",
                            precision=0,
                            interactive=False,
                        )
                        face_height = gr.Number(
                            value=config.model_config.face_default_height,
                            label="Height",
                            precision=0,
                            interactive=False,
                        )
                        face_batch_size = gr.Number(
                            value=1,
                            label="Batch",
                            precision=0,
                            interactive=False,
                        )
                        face_seed = gr.Number(value=-1, label="Seed  (−1 = random)", precision=0)

                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, elem_classes=["bv-output-card"]):
                        face_image_output = gr.Image(
                            type="pil",
                            label="Generated Face Reference Image",
                            height=512,
                        )
                    with gr.Column(scale=2, elem_classes=["bv-info-card"]):
                        face_info_text = gr.Textbox(
                            label="Generation Info",
                            interactive=False,
                            lines=8,
                        )

            # ── FOOTER ────────────────────────────────────────────────
            gr.HTML(
                "<footer class='bv-footer'>"
                "<div class='bv-footer-inner'>"
                "<div class='bv-footer-left'>"
                "<div class='bv-footer-brand'>Butter<span>Vision</span></div>"
                "<div class='bv-footer-tagline'>AI Image Generation · Stable Diffusion</div>"
                "</div>"
                "<div class='bv-footer-right'>"
                "<div class='bv-footer-tech'>"
                "<span class='bv-footer-badge'>Gradio</span>"
                "<span class='bv-footer-badge'>Diffusers</span>"
                "<span class='bv-footer-badge'>PyTorch</span>"
                "</div>"
                "<div class='bv-footer-copy'>© 2024 ButterVision</div>"
                "</div>"
                "</div>"
                "</footer>"
            )

            # ── MODAL (Fuera del footer) ──────────────────────────────
            gr.HTML(
                "<div id='bv-image-modal' class='bv-image-modal'>"
                "<div class='bv-modal-shell'>"
                "<div class='bv-modal-bar'>"
                "<div class='bv-modal-title'>Generated Image</div>"
                "<button class='bv-modal-close' data-bv-modal-close type='button'>×</button>"
                "</div>"
                "<div class='bv-modal-image-wrap'>"
                "<img id='bv-modal-image' alt='Generated image preview' />"
                "</div>"
                "</div>"
                "</div>"
            )

            # Inyectar JS mediante HTML para asegurar ejecución
            gr.HTML(f"<script>{UI_JS}</script>", visible=False)

            # ── EVENT HANDLERS ───────────────────────────────────────────
            function_selector.change(
                fn=self.switch_function,
                inputs=[function_selector],
                outputs=[text_to_image_panel, face_reference_panel, face_info_text],
            )

            generate_btn.click(
                fn=self.txt2img_generate,
                inputs=[prompt, negative_prompt, steps, cfg_scale, width, height, seed, batch_size],
                outputs=[image_output, info_text, seed, history_gallery],
            )
            face_generate_btn.click(
                fn=self.face_reference_generate,
                inputs=[
                    face_reference_image,
                    face_prompt,
                    face_negative_prompt,
                    face_steps,
                    face_cfg_scale,
                    face_width,
                    face_height,
                    face_seed,
                    face_batch_size,
                    face_identity_strength,
                    face_structure_strength,
                ],
                outputs=[face_image_output, face_info_text, face_seed, history_gallery],
            )
            
            # Abrir modal cuando la imagen cambia
            image_output.change(
                fn=None,
                inputs=[image_output],
                js="(url) => { if (url && window.openBVModal) window.openBVModal(url); }"
            )
            face_image_output.change(
                fn=None,
                inputs=[face_image_output],
                js="(url) => { if (url && window.openBVModal) window.openBVModal(url); }"
            )

            model_selector.change(
                fn=self.select_model,
                inputs=[model_selector],
                outputs=[model_status],
            )
            refresh_models_btn.click(
                fn=self.refresh_models,
                inputs=[],
                outputs=[model_selector, model_status],
            )
            refresh_history_btn.click(
                fn=self.refresh_history,
                inputs=[],
                outputs=[history_gallery],
            )

        return interface


def create_ui():
    """Crea la interfaz de ButterVision."""
    ui = ButterVisionUI()
    return ui.create_interface()


def get_ui_css():
    """CSS de la interfaz."""
    return UI_CSS


def get_ui_js():
    """JS de la interfaz."""
    return UI_JS
