package com.google.mediapipe.examples.handlandmarker.gl

import android.opengl.*
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Minimal EGL/GLES2 converter: YUV_420_888 (Y + interleaved UV) -> RGBA8888.
 * - Uploads Y as GL_LUMINANCE, UV as GL_LUMINANCE_ALPHA (U in .r, V in .a)
 * - Renders to an offscreen FBO, then glReadPixels to a reused RGBA ByteBuffer.
 *
 * Note: We must pack the UV plane to contiguous interleaved UV. If the incoming
 * ImageProxy has pixelStride != 2 or rowStride != width, we repack to a small
 * buffer (width/2 * height/2 * 2).
 */
class YuvGlConverter {
    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglSurface: EGLSurface? = null

    private var program = 0
    private var aPosLoc = 0
    private var aTexLoc = 0
    private var ySamplerLoc = 0
    private var uvSamplerLoc = 0

    private var yTex = 0
    private var uvTex = 0
    private var fbo = 0
    private var fboTex = 0

    private var width = 0
    private var height = 0

    private var rgbaBuffer: ByteBuffer? = null
    private var uvPackedBuffer: ByteBuffer? = null

    private val quadCoords: FloatBuffer = floatBufferOf(
        // x, y,    u, v
        -1f, -1f,   0f, 1f,
         1f, -1f,   1f, 1f,
        -1f,  1f,   0f, 0f,
         1f,  1f,   1f, 0f
    )

    fun ensureInitialized(w: Int, h: Int) {
        if (eglDisplay != null && w == width && h == height) return
        tearDown()
        width = w
        height = h
        setupEgl()
        setupGl()
        setupFbo()
    }

    fun convertToRgba(image: ImageProxy): ByteBuffer {
        val w = image.width
        val h = image.height
        ensureInitialized(w, h)

        // Prepare Y buffer contiguous (width x height)
        val yPlane = image.planes[0]
        val ySrc = yPlane.buffer
        val yRowStride = yPlane.rowStride
        val yPixStride = yPlane.pixelStride
        val yContig = ensureCapacityRgbaBuffer(w * h, forRgba = false)
        yContig.clear()
        // Copy Y plane using absolute indexing to avoid position/limit issues
        for (rowIdx in 0 until h) {
            val base = rowIdx * yRowStride
            if (yPixStride == 1) {
                var col = 0
                while (col < w) {
                    yContig.put(ySrc.get(base + col))
                    col++
                }
            } else {
                var col = 0
                while (col < w) {
                    yContig.put(ySrc.get(base + col * yPixStride))
                    col++
                }
            }
        }
        yContig.rewind()

        // Prepare UV interleaved (w/2 x h/2 x 2): pack U then V
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val uvW = w / 2
        val uvH = h / 2
        val uvBuf = ensureCapacityRgbaBuffer(uvW * uvH * 2, forRgba = false)
        uvBuf.clear()

        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixStride = uPlane.pixelStride
        val vPixStride = vPlane.pixelStride
        val uSrc = uPlane.buffer
        val vSrc = vPlane.buffer
        uSrc.rewind()
        vSrc.rewind()

        for (rowIdx in 0 until uvH) {
            val uBase = rowIdx * uRowStride
            val vBase = rowIdx * vRowStride
            var i = 0
            while (i < uvW) {
                uvBuf.put(uSrc.get(uBase + i * uPixStride))
                uvBuf.put(vSrc.get(vBase + i * vPixStride))
                i++
            }
        }
        uvBuf.rewind()

        // Upload textures
        GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, yTex)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE, w, h, 0,
            GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, yContig
        )

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, uvTex)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE_ALPHA, uvW, uvH, 0,
            GLES20.GL_LUMINANCE_ALPHA, GLES20.GL_UNSIGNED_BYTE, uvBuf
        )

        // Render to FBO
        GLES20.glViewport(0, 0, w, h)
        GLES20.glUseProgram(program)
        GLES20.glUniform1i(ySamplerLoc, 0)
        GLES20.glUniform1i(uvSamplerLoc, 1)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)

        quadCoords.position(0)
        GLES20.glEnableVertexAttribArray(aPosLoc)
        GLES20.glVertexAttribPointer(aPosLoc, 2, GLES20.GL_FLOAT, false, 16, quadCoords)
        quadCoords.position(2)
        GLES20.glEnableVertexAttribArray(aTexLoc)
        GLES20.glVertexAttribPointer(aTexLoc, 2, GLES20.GL_FLOAT, false, 16, quadCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        // Read back RGBA
        val out = ensureCapacityRgbaBuffer(w * h * 4, forRgba = true)
        out.clear()
        GLES20.glReadPixels(0, 0, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, out)
        out.rewind()
        return out
    }

    fun release() {
        tearDown()
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
        program = buildProgram(VERT, FRAG)
        aPosLoc = GLES20.glGetAttribLocation(program, "aPosition")
        aTexLoc = GLES20.glGetAttribLocation(program, "aTexCoord")
        ySamplerLoc = GLES20.glGetUniformLocation(program, "uTexY")
        uvSamplerLoc = GLES20.glGetUniformLocation(program, "uTexUV")

        yTex = genTexture()
        uvTex = genTexture()
    }

    private fun setupFbo() {
        val t = IntArray(1)
        GLES20.glGenFramebuffers(1, t, 0)
        fbo = t[0]

        val texArr = IntArray(1)
        GLES20.glGenTextures(1, texArr, 0)
        fboTex = texArr[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTex)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null
        )
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo)
        GLES20.glFramebufferTexture2D(
            GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
            GLES20.GL_TEXTURE_2D, fboTex, 0
        )
        check(GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER) == GLES20.GL_FRAMEBUFFER_COMPLETE)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
    }

    private fun tearDown() {
        try {
            if (program != 0) GLES20.glDeleteProgram(program)
        } catch (_: Exception) {}
        program = 0
        aPosLoc = 0; aTexLoc = 0; ySamplerLoc = 0; uvSamplerLoc = 0
        try { if (yTex != 0) GLES20.glDeleteTextures(1, intArrayOf(yTex), 0) } catch (_: Exception) {}
        try { if (uvTex != 0) GLES20.glDeleteTextures(1, intArrayOf(uvTex), 0) } catch (_: Exception) {}
        try { if (fboTex != 0) GLES20.glDeleteTextures(1, intArrayOf(fboTex), 0) } catch (_: Exception) {}
        try { if (fbo != 0) GLES20.glDeleteFramebuffers(1, intArrayOf(fbo), 0) } catch (_: Exception) {}
        yTex = 0; uvTex = 0; fbo = 0; fboTex = 0

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
        eglDisplay = null
        eglSurface = null
        eglContext = null
        rgbaBuffer = null
        uvPackedBuffer = null
    }

    private fun buildProgram(vSrc: String, fSrc: String): Int {
        val vs = compile(GLES20.GL_VERTEX_SHADER, vSrc)
        val fs = compile(GLES20.GL_FRAGMENT_SHADER, fSrc)
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, vs)
        GLES20.glAttachShader(prog, fs)
        GLES20.glLinkProgram(prog)
        val link = IntArray(1)
        GLES20.glGetProgramiv(prog, GLES20.GL_LINK_STATUS, link, 0)
        if (link[0] == 0) {
            val log = GLES20.glGetProgramInfoLog(prog)
            GLES20.glDeleteProgram(prog)
            throw RuntimeException("Program link failed: $log")
        }
        GLES20.glDeleteShader(vs)
        GLES20.glDeleteShader(fs)
        return prog
    }

    private fun compile(type: Int, src: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, src)
        GLES20.glCompileShader(shader)
        val ok = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            throw RuntimeException("Shader compile failed: $log")
        }
        return shader
    }

    private fun genTexture(): Int {
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        val id = t[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return id
    }

    private fun ensureCapacityRgbaBuffer(capacity: Int, forRgba: Boolean): ByteBuffer {
        return if (forRgba) {
            if (rgbaBuffer == null || rgbaBuffer!!.capacity() < capacity) {
                rgbaBuffer = ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder())
            }
            rgbaBuffer!!
        } else {
            if (uvPackedBuffer == null || uvPackedBuffer!!.capacity() < capacity) {
                uvPackedBuffer = ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder())
            }
            uvPackedBuffer!!
        }
    }

    companion object {
        private const val VERT = """
            attribute vec4 aPosition;
            attribute vec2 aTexCoord;
            varying vec2 vTexCoord;
            void main() {
              gl_Position = aPosition;
              vTexCoord = aTexCoord;
            }
        """

        private const val FRAG = """
            precision mediump float;
            varying vec2 vTexCoord;
            uniform sampler2D uTexY;
            uniform sampler2D uTexUV;
            void main() {
              float y = texture2D(uTexY, vTexCoord).r;
              vec2 uv = texture2D(uTexUV, vTexCoord).ra - vec2(0.5, 0.5);
              float r = y + 1.402 * uv.y;
              float g = y - 0.344136 * uv.x - 0.714136 * uv.y;
              float b = y + 1.772 * uv.x;
              gl_FragColor = vec4(r, g, b, 1.0);
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
