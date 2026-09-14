"""Fail-closed OpenGL provenance for MuJoCo's main offscreen framebuffer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from epsbench.diagnostics.capture import DiagnosticFailure


def _scalar(value: object) -> int:
    return int(value[0] if hasattr(value, "__len__") else value)


def _integer(gl: Any, constant: object) -> int:
    return _scalar(gl.glGetIntegerv(constant))


def _attachment(gl: Any, target: object, attachment: object) -> Mapping[str, object]:
    object_type = _scalar(
        gl.glGetFramebufferAttachmentParameteriv(
            target, attachment, gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE
        )
    )
    object_name = _scalar(
        gl.glGetFramebufferAttachmentParameteriv(
            target, attachment, gl.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME
        )
    )
    component_type = _scalar(
        gl.glGetFramebufferAttachmentParameteriv(
            target, attachment, gl.GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE
        )
    )
    result: dict[str, object] = {
        "object_type": object_type,
        "object_name": object_name,
        "component_type": component_type,
    }
    if object_type == int(gl.GL_RENDERBUFFER):
        previous = _integer(gl, gl.GL_RENDERBUFFER_BINDING)
        try:
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, object_name)
            result["internal_format"] = _integer(gl, gl.GL_RENDERBUFFER_INTERNAL_FORMAT)
            result["samples"] = _integer(gl, gl.GL_RENDERBUFFER_SAMPLES)
        finally:
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, previous)
    return result


def inspect_mujoco_offscreen_attachments(renderer: object) -> Mapping[str, object]:
    """Inspect main offFBO and optional resolve offFBO_r without leaking bindings."""
    try:
        from OpenGL import GL
    except ImportError as exc:
        raise DiagnosticFailure("PyOpenGL is required for GL attachment provenance") from exc
    context = getattr(renderer, "_mjr_context", None)
    if context is None:
        raise DiagnosticFailure("MuJoCo renderer exposes no offscreen context")
    off_fbo = getattr(context, "offFBO", None)
    if off_fbo is None:
        raise DiagnosticFailure("MuJoCo context exposes no main offFBO")
    saved_read = _integer(GL, GL.GL_READ_FRAMEBUFFER_BINDING)
    saved_draw = _integer(GL, GL.GL_DRAW_FRAMEBUFFER_BINDING)
    saved_renderbuffer = _integer(GL, GL.GL_RENDERBUFFER_BINDING)
    records: dict[str, object] = {}
    try:
        for name in ("offFBO", "offFBO_r"):
            fbo = getattr(context, name, None)
            if fbo is None or int(fbo) == 0:
                records[name] = {"present": False}
                continue
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(fbo))
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(fbo))
            records[name] = {
                "present": True,
                "framebuffer": int(fbo),
                "color0": dict(_attachment(GL, GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0)),
            }
    finally:
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, saved_read)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, saved_draw)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, saved_renderbuffer)
    main = records["offFBO"]
    if not isinstance(main, dict) or not main.get("present"):
        raise DiagnosticFailure("main offFBO is unavailable")
    color = main.get("color0")
    if not isinstance(color, dict):
        raise DiagnosticFailure("main offFBO color attachment is unavailable")
    return {
        "attachment_format": color.get("internal_format"),
        "attachment_component_type": color.get("component_type"),
        "offscreen_attachments": records,
    }
