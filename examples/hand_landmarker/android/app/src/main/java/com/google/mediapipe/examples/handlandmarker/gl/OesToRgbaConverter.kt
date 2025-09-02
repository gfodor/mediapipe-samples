package com.google.mediapipe.examples.handlandmarker.gl

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.GLES11Ext
import android.opengl.GLES20
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class OesToRgbaConverter(private val width: Int, private val height: Int) {
    private var eglDisplay: android.opengl.EGLDisplay? = null
    private var eglContext: android.opengl.EGLContext? = null
    private var eglSurface: android.opengl.EGLSurface? = null

    private var program = 0
    private var aPosLoc = 0
    private var aTexLoc = 0
    private var uTexLoc = 0

    private var fbo = 0
    private var fboTex = 0
    private var oesTex = 0

    private var rgbaBuffer: ByteBuffer? = null

    private val quad: FloatBuffer = floatBufferOf(
        -1f, -1f, 0f, 1f,
         1f, -1f, 1f, 1f,
        -1f,  1f, 0f, 0f,
         1f,  1f, 1f, 0f
    )

    init {
        setupEgl()
        setupGl()
        setupFbo()
    }

    fun getExternalTextureId(): Int = oesTex

    fun convert(frame: Frame): ByteBuffer {
        // Draw OES to FBO
        GLES20.glViewport(0, 0, width, height)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES20.glUniform1i(uTexLoc, 0)

        quad.position(0)
        GLES20.glEnableVertexAttribArray(aPosLoc)
        GLES20.glVertexAttribPointer(aPosLoc, 2, GLES20.GL_FLOAT, false, 16, quad)
        quad.position(2)
        GLES20.glEnableVertexAttribArray(aTexLoc)
        GLES20.glVertexAttribPointer(aTexLoc, 2, GLES20.GL_FLOAT, false, 16, quad)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        val out = ensureRgbaBuffer(width * height * 4)
        out.clear()
        GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, out)
        out.rewind()
        return out
    }

    fun release() {
        try { if (program != 0) GLES20.glDeleteProgram(program) } catch (_: Exception) {}
        try { if (fboTex != 0) GLES20.glDeleteTextures(1, intArrayOf(fboTex), 0) } catch (_: Exception) {}
        try { if (oesTex != 0) GLES20.glDeleteTextures(1, intArrayOf(oesTex), 0) } catch (_: Exception) {}
        try { if (fbo != 0) GLES20.glDeleteFramebuffers(1, intArrayOf(fbo), 0) } catch (_: Exception) {}
        if (eglDisplay != null) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
            if (eglSurface != null && eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface)
            }
            if (eglContext != null && eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext)
            }
            EGL14.eglReleaseThread()
            EGL14.eglTerminate(eglDisplay)
        }
        eglDisplay = null; eglContext = null; eglSurface = null
        rgbaBuffer = null
    }

    private fun setupEgl() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        val vers = IntArray(2)
        check(EGL14.eglInitialize(eglDisplay, vers, 0, vers, 1))

        val cfgAttrs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numCfg = IntArray(1)
        check(EGL14.eglChooseConfig(eglDisplay, cfgAttrs, 0, configs, 0, 1, numCfg, 0))
        val cfg = configs[0]

        val ctxAttrs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
        eglContext = EGL14.eglCreateContext(eglDisplay, cfg, EGL14.EGL_NO_CONTEXT, ctxAttrs, 0)
        check(eglContext != null && eglContext != EGL14.EGL_NO_CONTEXT)

        val surfAttrs = intArrayOf(
            EGL14.EGL_WIDTH, width,
            EGL14.EGL_HEIGHT, height,
            EGL14.EGL_NONE
        )
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, cfg, surfAttrs, 0)
        check(eglSurface != null && eglSurface != EGL14.EGL_NO_SURFACE)
        check(EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext))
    }

    private fun setupGl() {
        program = buildProgram(VS, FS)
        aPosLoc = GLES20.glGetAttribLocation(program, "aPosition")
        aTexLoc = GLES20.glGetAttribLocation(program, "aTexCoord")
        uTexLoc = GLES20.glGetUniformLocation(program, "uTex")

        // External OES texture for ARCore camera
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        oesTex = t[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTex)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
    }

    private fun setupFbo() {
        val fb = IntArray(1)
        GLES20.glGenFramebuffers(1, fb, 0)
        fbo = fb[0]

        val tex = IntArray(1)
        GLES20.glGenTextures(1, tex, 0)
        fboTex = tex[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTex)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, fboTex, 0)
        check(GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) == GLES20.GL_FRAMEBUFFER_COMPLETE)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
    }

    private fun ensureRgbaBuffer(capacity: Int): ByteBuffer {
        if (rgbaBuffer == null || rgbaBuffer!!.capacity() < capacity) {
            rgbaBuffer = ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder())
        }
        return rgbaBuffer!!
    }

    private fun buildProgram(vsrc: String, fsrc: String): Int {
        val vs = compile(GLES20.GL_VERTEX_SHADER, vsrc)
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fsrc)
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, vs)
        GLES20.glAttachShader(prog, fs)
        GLES20.glLinkProgram(prog)
        val ok = IntArray(1)
        GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, ok, 0)
        if (ok[0] == 0) throw RuntimeException("link: " + GLES20.glGetProgramInfoLog(prog))
        GLES20.glDeleteShader(vs)
        GLES20.glDeleteShader(fs)
        return prog
    }

    private fun compile(type: Int, src: String): Int {
        val id = GLES20.glCreateShader(type)
        GLES20.glShaderSource(id, src)
        GLES20.glCompileShader(id)
        val ok = IntArray(1)
        GLES20.glGetShaderiv(id, GLES20.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) throw RuntimeException("compile: " + GLES20.glGetShaderInfoLog(id))
        return id
    }

    companion object {
        private const val VS = """
            attribute vec4 aPosition;
            attribute vec2 aTexCoord;
            varying vec2 vTexCoord;
            void main() {
              gl_Position = aPosition;
              vTexCoord = aTexCoord;
            }
        """

        private const val FS = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 vTexCoord;
            uniform samplerExternalOES uTex;
            void main() {
              vec4 c = texture2D(uTex, vTexCoord);
              gl_FragColor = vec4(c.rgb, 1.0);
            }
        """

        private fun floatBufferOf(vararg v: Float): FloatBuffer {
            val bb = ByteBuffer.allocateDirect(v.size * 4).order(ByteOrder.nativeOrder())
            val fb = bb.asFloatBuffer()
            fb.put(v)
            fb.position(0)
            return fb
        }
    }
}

