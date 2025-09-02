/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.handlandmarker.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.content.Context
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import android.graphics.SurfaceTexture
import android.graphics.ImageFormat
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import android.hardware.camera2.CameraCharacteristics
import com.google.mediapipe.examples.handlandmarker.HandLandmarkerHelper
import com.google.mediapipe.examples.handlandmarker.MainViewModel
import com.google.mediapipe.examples.handlandmarker.R
import com.google.mediapipe.examples.handlandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.components.processors.ClassifierOptions
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.examples.handlandmarker.gl.YuvGlConverter
import com.google.mediapipe.examples.handlandmarker.gl.OesToRgbaConverter
import com.google.ar.core.Session
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Frame
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class CameraFragment : Fragment(), HandLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Hand Landmarker"
        private const val PREFS_NAME = "mp_hand_landmarker_settings"
        private const val KEY_DELEGATE = "delegate"
        private const val KEY_DETECTION = "min_hand_detection_conf"
        private const val KEY_TRACKING = "min_hand_tracking_conf"
        private const val KEY_PRESENCE = "min_hand_presence_conf"
        private const val KEY_MAX_HANDS = "max_hands"
        private const val KEY_GESTURE_THRESHOLD = "gesture_threshold"
        private const val KEY_PINCH_THRESHOLD = "pinch_threshold"
        private const val KEY_PINCH_RELEASE_THRESHOLD = "pinch_release_threshold"
        private const val KEY_RES_W = "res_width"
        private const val KEY_RES_H = "res_height"
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var handLandmarkerHelper: HandLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK
    private var gestureThreshold: Float = 0.1f
    private var pinchThreshold: Float = 2.0f
    private var pinchReleaseThreshold: Float = 3.0f
    @Volatile private var isPinching: Boolean = false
    private var availableResolutions: List<Size> = emptyList()
    private var selectedResolution: Size? = null
    private var gestureRecognizer: GestureRecognizer? = null
    @Volatile private var lastInputWidth: Int = 0
    @Volatile private var lastInputHeight: Int = 0
    // Reusable RGBA bitmap to avoid per-frame allocations
    private var rgbaBitmap: android.graphics.Bitmap? = null
    private var rgbaW: Int = 0
    private var rgbaH: Int = 0
    private var yuvGl: YuvGlConverter? = null
    private var arSession: Session? = null
    private var oesConv: OesToRgbaConverter? = null
    @Volatile private var arRunning = false
    private val targetW = 640
    private val targetH = 480
    @Volatile private var arInstallRequested = true

    private fun getPrefs() = requireContext().getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private fun loadSettings() {
        val p = getPrefs()
        viewModel.setMinHandDetectionConfidence(
            p.getFloat(KEY_DETECTION, viewModel.currentMinHandDetectionConfidence)
        )
        viewModel.setMinHandTrackingConfidence(
            p.getFloat(KEY_TRACKING, viewModel.currentMinHandTrackingConfidence)
        )
        viewModel.setMinHandPresenceConfidence(
            p.getFloat(KEY_PRESENCE, viewModel.currentMinHandPresenceConfidence)
        )
        viewModel.setMaxHands(p.getInt(KEY_MAX_HANDS, viewModel.currentMaxHands))
        viewModel.setDelegate(p.getInt(KEY_DELEGATE, viewModel.currentDelegate))
        gestureThreshold = p.getFloat(KEY_GESTURE_THRESHOLD, 0.1f)
        pinchThreshold = p.getFloat(KEY_PINCH_THRESHOLD, 2.0f)
        pinchReleaseThreshold = p.getFloat(KEY_PINCH_RELEASE_THRESHOLD, 3.0f)
        val rw = p.getInt(KEY_RES_W, -1)
        val rh = p.getInt(KEY_RES_H, -1)
        selectedResolution = if (rw > 0 && rh > 0) Size(rw, rh) else null
    }
    private fun saveSettings() {
        try {
            val editor = getPrefs().edit()
            if (this::handLandmarkerHelper.isInitialized) {
                editor.putInt(KEY_DELEGATE, handLandmarkerHelper.currentDelegate)
                editor.putFloat(KEY_DETECTION, handLandmarkerHelper.minHandDetectionConfidence)
                editor.putFloat(KEY_TRACKING, handLandmarkerHelper.minHandTrackingConfidence)
                editor.putFloat(KEY_PRESENCE, handLandmarkerHelper.minHandPresenceConfidence)
                editor.putInt(KEY_MAX_HANDS, handLandmarkerHelper.maxNumHands)
            } else {
                editor.putInt(KEY_DELEGATE, viewModel.currentDelegate)
                editor.putFloat(KEY_DETECTION, viewModel.currentMinHandDetectionConfidence)
                editor.putFloat(KEY_TRACKING, viewModel.currentMinHandTrackingConfidence)
                editor.putFloat(KEY_PRESENCE, viewModel.currentMinHandPresenceConfidence)
                editor.putInt(KEY_MAX_HANDS, viewModel.currentMaxHands)
            }
            editor.putFloat(KEY_GESTURE_THRESHOLD, gestureThreshold)
            editor.putFloat(KEY_PINCH_THRESHOLD, pinchThreshold)
            editor.putFloat(KEY_PINCH_RELEASE_THRESHOLD, pinchReleaseThreshold)
            selectedResolution?.let { s ->
                editor.putInt(KEY_RES_W, s.width)
                editor.putInt(KEY_RES_H, s.height)
            }
            editor.apply()
        } catch (_: Exception) {
            // ignore
        }
    }

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
        }

        // Start the HandLandmarkerHelper again when users come back
        // to the foreground.
        backgroundExecutor.execute {
            if (handLandmarkerHelper.isClose()) {
                handLandmarkerHelper.setupHandLandmarker()
            }
            if (gestureRecognizer == null) {
                setupGestureRecognizer()
            }
        }
        // Start ARCore on UI thread to satisfy lifecycle expectations
        activity?.runOnUiThread { startArLoop() }
    }

    override fun onPause() {
        super.onPause()
        if(this::handLandmarkerHelper.isInitialized) {
            viewModel.setMaxHands(handLandmarkerHelper.maxNumHands)
            viewModel.setMinHandDetectionConfidence(handLandmarkerHelper.minHandDetectionConfidence)
            viewModel.setMinHandTrackingConfidence(handLandmarkerHelper.minHandTrackingConfidence)
            viewModel.setMinHandPresenceConfidence(handLandmarkerHelper.minHandPresenceConfidence)
            viewModel.setDelegate(handLandmarkerHelper.currentDelegate)

            // Close the HandLandmarkerHelper and GestureRecognizer; release resources
            // Pause ARCore on UI thread, then release others in background
            activity?.runOnUiThread { stopArLoop() }
            backgroundExecutor.execute {
                handLandmarkerHelper.clearHandLandmarker()
                closeGestureRecognizer()
                try { yuvGl?.release() } catch (_: Exception) {}
            }
            saveSettings()
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(
            Long.MAX_VALUE, TimeUnit.NANOSECONDS
        )
        try { yuvGl?.release() } catch (_: Exception) {}
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // Load persisted settings before creating helper
        loadSettings()

        // Ensure gesture label overlays above preview/overlay
        fragmentCameraBinding.gestureLabel.bringToFront()
        try { fragmentCameraBinding.gestureLabel.translationZ = 1000f } catch (_: Exception) {}
        fragmentCameraBinding.pinchLabel.bringToFront()
        try { fragmentCameraBinding.pinchLabel.translationZ = 1000f } catch (_: Exception) {}

        // Hide CameraX view and set a white background. ARCore provides frames offscreen.
        fragmentCameraBinding.viewFinder.visibility = View.GONE
        fragmentCameraBinding.cameraContainer.setBackgroundColor(android.graphics.Color.WHITE)

        // Create the HandLandmarkerHelper that will handle the inference
        backgroundExecutor.execute {
            handLandmarkerHelper = HandLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minHandDetectionConfidence = viewModel.currentMinHandDetectionConfidence,
                minHandTrackingConfidence = viewModel.currentMinHandTrackingConfidence,
                minHandPresenceConfidence = viewModel.currentMinHandPresenceConfidence,
                maxNumHands = viewModel.currentMaxHands,
                currentDelegate = viewModel.currentDelegate,
                handLandmarkerHelperListener = this
            )
            // Initialize built-in gesture recognizer on the same executor
            setupGestureRecognizer()
            saveSettings()
        }

        // Attach listeners to UI control widgets
        initBottomSheetControls()
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings
        fragmentCameraBinding.bottomSheetLayout.maxHandsValue.text =
            viewModel.currentMaxHands.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinHandPresenceConfidence
            )

        // Gesture threshold
        fragmentCameraBinding.bottomSheetLayout.gestureThresholdValue.text =
            String.format(Locale.US, "%.3f", gestureThreshold)

        // Pinch threshold
        fragmentCameraBinding.bottomSheetLayout.pinchThresholdValue.text =
            String.format(Locale.US, "%.2f", pinchThreshold)

        // Pinch release threshold
        fragmentCameraBinding.bottomSheetLayout.pinchReleaseThresholdValue.text =
            String.format(Locale.US, "%.2f", pinchReleaseThreshold)

        // When clicked, lower hand detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandDetectionConfidence >= 0.2) {
                handLandmarkerHelper.minHandDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandDetectionConfidence <= 0.8) {
                handLandmarkerHelper.minHandDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower hand tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandTrackingConfidence >= 0.2) {
                handLandmarkerHelper.minHandTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandTrackingConfidence <= 0.8) {
                handLandmarkerHelper.minHandTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower hand presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (handLandmarkerHelper.minHandPresenceConfidence >= 0.2) {
                handLandmarkerHelper.minHandPresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise hand presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (handLandmarkerHelper.minHandPresenceConfidence <= 0.8) {
                handLandmarkerHelper.minHandPresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // Pinch threshold controls
        fragmentCameraBinding.bottomSheetLayout.pinchThresholdMinus.setOnClickListener {
            // Reasonable bounds 0.5..5.0, step 0.05
            if (pinchThreshold > 0.5f) {
                pinchThreshold = (pinchThreshold - 0.05f).coerceAtLeast(0.5f)
                fragmentCameraBinding.bottomSheetLayout.pinchThresholdValue.text =
                    String.format(Locale.US, "%.2f", pinchThreshold)
                saveSettings()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.pinchThresholdPlus.setOnClickListener {
            if (pinchThreshold < 10.0f) {
                pinchThreshold = (pinchThreshold + 0.05f).coerceAtMost(10.0f)
                fragmentCameraBinding.bottomSheetLayout.pinchThresholdValue.text =
                    String.format(Locale.US, "%.2f", pinchThreshold)
                saveSettings()
            }
        }

        // Pinch release threshold controls
        fragmentCameraBinding.bottomSheetLayout.pinchReleaseThresholdMinus.setOnClickListener {
            if (pinchReleaseThreshold > 0.5f) {
                pinchReleaseThreshold = (pinchReleaseThreshold - 0.05f).coerceAtLeast(0.5f)
                fragmentCameraBinding.bottomSheetLayout.pinchReleaseThresholdValue.text =
                    String.format(Locale.US, "%.2f", pinchReleaseThreshold)
                saveSettings()
            }
        }
        fragmentCameraBinding.bottomSheetLayout.pinchReleaseThresholdPlus.setOnClickListener {
            if (pinchReleaseThreshold < 10.0f) {
                pinchReleaseThreshold = (pinchReleaseThreshold + 0.05f).coerceAtMost(10.0f)
                fragmentCameraBinding.bottomSheetLayout.pinchReleaseThresholdValue.text =
                    String.format(Locale.US, "%.2f", pinchReleaseThreshold)
                saveSettings()
            }
        }

        // When clicked, reduce the number of hands that can be detected at a
        // time
        fragmentCameraBinding.bottomSheetLayout.maxHandsMinus.setOnClickListener {
            if (handLandmarkerHelper.maxNumHands > 1) {
                handLandmarkerHelper.maxNumHands--
                updateControlsUi()
            }
        }

        // When clicked, increase the number of hands that can be detected
        // at a time
        fragmentCameraBinding.bottomSheetLayout.maxHandsPlus.setOnClickListener {
            if (handLandmarkerHelper.maxNumHands < 2) {
                handLandmarkerHelper.maxNumHands++
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        handLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "HandLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }

        // Gesture threshold minus
        fragmentCameraBinding.bottomSheetLayout.gestureThresholdMinus.setOnClickListener {
            if (gestureThreshold >= 0.005f) {
                gestureThreshold = (gestureThreshold - 0.005f).coerceAtLeast(0f)
                updateControlsUi()
            }
        }

        // Gesture threshold plus
        fragmentCameraBinding.bottomSheetLayout.gestureThresholdPlus.setOnClickListener {
            if (gestureThreshold <= 0.995f) {
                gestureThreshold = (gestureThreshold + 0.005f).coerceAtMost(1f)
                updateControlsUi()
            }
        }
    }

    // Update the values displayed in the bottom sheet. Reset Handlandmarker
    // helper.
    private fun updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxHandsValue.text =
            handLandmarkerHelper.maxNumHands.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                handLandmarkerHelper.minHandPresenceConfidence
            )

        // Update gesture threshold UI
        fragmentCameraBinding.bottomSheetLayout.gestureThresholdValue.text =
            String.format(Locale.US, "%.3f", gestureThreshold)
        fragmentCameraBinding.bottomSheetLayout.pinchThresholdValue.text =
            String.format(Locale.US, "%.2f", pinchThreshold)
        fragmentCameraBinding.bottomSheetLayout.pinchReleaseThresholdValue.text =
            String.format(Locale.US, "%.2f", pinchReleaseThreshold)

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        backgroundExecutor.execute {
            
            handLandmarkerHelper.clearHandLandmarker()
            handLandmarkerHelper.setupHandLandmarker()
            closeGestureRecognizer()
            setupGestureRecognizer()
            saveSettings()
        }
        fragmentCameraBinding.overlay.clear()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        // no-op; replaced by ARCore input
    }

    private fun findWidestBackCameraInfo(): androidx.camera.core.CameraInfo? {
        val provider = cameraProvider ?: return null
        var bestInfo: androidx.camera.core.CameraInfo? = null
        var minFocal = Float.MAX_VALUE
        for (info in provider.availableCameraInfos) {
            try {
                val c2 = Camera2CameraInfo.from(info)
                val facing = c2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
                if (facing != CameraCharacteristics.LENS_FACING_BACK) continue
                val focals = c2.getCameraCharacteristic(
                    CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS
                )
                if (focals == null || focals.isEmpty()) continue
                var f = Float.MAX_VALUE
                for (v in focals) if (v < f) f = v
                if (f < minFocal) {
                    minFocal = f
                    bestInfo = info
                }
            } catch (_: Exception) { }
        }
        return bestInfo
    }

    private fun setupResolutionSpinner() {
        val info = findWidestBackCameraInfo() ?: return
        val chars = Camera2CameraInfo.from(info)
            .getCameraCharacteristic(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return
        val previewSizes = chars.getOutputSizes(SurfaceTexture::class.java)?.toList() ?: emptyList()
        val analysisSizes = chars.getOutputSizes(ImageFormat.YUV_420_888)?.toList() ?: emptyList()
        val intersect = previewSizes.intersect(analysisSizes.toSet()).toList()
        if (intersect.isEmpty()) return
        availableResolutions = intersect.sortedByDescending { it.width * it.height }

        // Pick default: saved, else largest 4:3, else largest
        val saved = selectedResolution
        val defaultIndex = when {
            saved != null -> availableResolutions.indexOfFirst { it.width == saved.width && it.height == saved.height }.takeIf { it >= 0 }
            else -> null
        } ?: run {
            val fourThirds = availableResolutions
                .withIndex()
                .filter { (_, s) -> kotlin.math.abs((s.width.toFloat() / s.height) - (4f/3f)) < 0.02f }
            if (fourThirds.isNotEmpty()) fourThirds.maxByOrNull { it.value.width * it.value.height }?.index ?: 0
            else 0
        }

        val entries = availableResolutions.map { "${it.width}x${it.height}" }
        val adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, entries)
        fragmentCameraBinding.bottomSheetLayout.spinnerResolution.adapter = adapter
        fragmentCameraBinding.bottomSheetLayout.spinnerResolution.setSelection(defaultIndex, false)
        selectedResolution = availableResolutions[defaultIndex]

        fragmentCameraBinding.bottomSheetLayout.spinnerResolution.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (position in availableResolutions.indices) {
                    val newSize = availableResolutions[position]
                    if (selectedResolution != newSize) {
                        selectedResolution = newSize
                        saveSettings()
                        bindCameraUseCases()
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) { /* no-op */ }
        }
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() { /* no-op */ }
    
    private fun buildBackWidestCameraSelector(): CameraSelector {
        val widestBackFilter = androidx.camera.core.CameraFilter { cameraInfos ->
            // Pick the back camera with the smallest focal length (widest FoV)
            var bestInfo: androidx.camera.core.CameraInfo? = null
            var minFocal = Float.MAX_VALUE
            for (info in cameraInfos) {
                try {
                    val c2 = Camera2CameraInfo.from(info)
                    val focals = c2.getCameraCharacteristic(
                        CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS
                    )
                    if (focals == null || focals.isEmpty()) continue
                    var f = Float.MAX_VALUE
                    for (v in focals) if (v < f) f = v
                    if (f < minFocal) {
                        minFocal = f
                        bestInfo = info
                    }
                } catch (_: Exception) {
                    // Ignore and continue
                }
            }
            if (bestInfo != null) listOf(bestInfo) else cameraInfos
        }

        return CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .addCameraFilter(widestBackFilter)
            .build()
    }

    private fun detectHand(imageProxy: ImageProxy) { /* no-op */ }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // no-op for ARCore input
    }

    // Update UI after hand have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: HandLandmarkerHelper.ResultBundle
    ) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                    String.format("%d ms", resultBundle.inferenceTime)

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    if (lastInputHeight > 0) lastInputHeight else resultBundle.inputImageHeight,
                    if (lastInputWidth > 0) lastInputWidth else resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )

                // Force a redraw
                fragmentCameraBinding.overlay.invalidate()

                // Pinch detection using knuckle-scaled distance (distance invariant) with hysteresis
                try {
                    val hr = resultBundle.results.firstOrNull()
                    var minRelative = Double.MAX_VALUE
                    if (hr != null && hr.landmarks().isNotEmpty()) {
                        for (hand in hr.landmarks()) {
                            if (hand.size >= 9) {
                                val thumbTip = hand[4]
                                val indexTip = hand[8]
                                val thumbIp = hand[3]
                                val indexDip = hand[7]

                                val dTip = euclideanDistance2D(thumbTip.x().toDouble(), thumbTip.y().toDouble(), indexTip.x().toDouble(), indexTip.y().toDouble())
                                val dThumb = euclideanDistance2D(thumbTip.x().toDouble(), thumbTip.y().toDouble(), thumbIp.x().toDouble(), thumbIp.y().toDouble())
                                val dIndex = euclideanDistance2D(indexTip.x().toDouble(), indexTip.y().toDouble(), indexDip.x().toDouble(), indexDip.y().toDouble())

                                val denom = 0.5 * (dThumb + dIndex)
                                if (denom > 1e-6) {
                                    val relative = (dTip * 10.0) / denom
                                    if (relative < minRelative) minRelative = relative
                                }
                            }
                        }
                    }
                    // Update hysteresis state
                    if (!isPinching && minRelative < pinchThreshold) {
                        isPinching = true
                    } else if (isPinching && minRelative > pinchReleaseThreshold) {
                        isPinching = false
                    }
                    val pinchLabel = fragmentCameraBinding.pinchLabel
                    if (isPinching) {
                        pinchLabel.visibility = View.VISIBLE
                        pinchLabel.alpha = 1f
                        pinchLabel.bringToFront()
                        try { pinchLabel.translationZ = 1000f } catch (_: Exception) {}
                    } else {
                        pinchLabel.visibility = View.GONE
                    }
                } catch (_: Exception) { }
            }
        }
    }

    private fun euclideanDistance2D(x1: Double, y1: Double, x2: Double, y2: Double): Double {
        val dx = x1 - x2
        val dy = y1 - y2
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private fun setupGestureRecognizer() {
        try {
            
            val base = BaseOptions.builder()
                .setModelAssetPath("gesture_recognizer.task")
                .setDelegate(Delegate.GPU)
                .build()

            val canned = ClassifierOptions.builder()
                // Use adjustable sensitivity
                .setScoreThreshold(gestureThreshold)
                // Support both naming variants just in case
                .setCategoryAllowlist(listOf("Closed_Fist", "Fist_Closed"))
                .build()

            val opts = GestureRecognizer.GestureRecognizerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(2)
                .setCannedGesturesClassifierOptions(canned)
                .setMinHandDetectionConfidence(0.55f)
                .setMinHandPresenceConfidence(0.55f)
                .setMinTrackingConfidence(0.55f)
                .setResultListener(this::onGestureResults)
                .build()

            gestureRecognizer = GestureRecognizer.createFromOptions(requireContext(), opts)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init GestureRecognizer: ${e.message}")
        }
    }

    private fun closeGestureRecognizer() {
        try { gestureRecognizer?.close() } catch (_: Exception) {}
        gestureRecognizer = null
    }

    private fun onGestureResults(result: GestureRecognizerResult, input: MPImage) {
        // Log all categories per-hand
        try {
            val all = result.gestures()
            var idx = 0
            for (hand in all) {
                // consume to avoid unused warnings
                if (hand.isNotEmpty()) { val dummyName = hand[0].categoryName() }
                idx++
            }
        } catch (_: Exception) { }

        // Determine top category across hands
        var topName: String? = null
        var topScore = 0f
        try {
            for (hand in result.gestures()) {
                if (hand.isNotEmpty()) {
                    val cat = hand[0]
                    if (cat.score() > topScore) {
                        topScore = cat.score()
                        topName = cat.categoryName()
                    }
                }
            }
        } catch (_: Exception) { }

        val isFist = topName == "Closed_Fist" || topName == "Fist_Closed"
        val passes = topScore >= gestureThreshold
        activity?.runOnUiThread {
            val label = fragmentCameraBinding.gestureLabel
            if (isFist && passes) {
                label.text = String.format(Locale.US, "Fist Closed (%.2f)", topScore)
                label.setTextColor(android.graphics.Color.RED)
                label.visibility = View.VISIBLE
                label.alpha = 1f
                label.bringToFront()
                try { label.translationZ = 1000f } catch (_: Exception) {}
            } else {
                label.text = ""
                label.visibility = View.INVISIBLE
                label.alpha = 1f
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            if (errorCode == HandLandmarkerHelper.GPU_ERROR) {
                fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                    HandLandmarkerHelper.DELEGATE_CPU, false
                )
            }
        }
    }

    // ARCore integration: offscreen session using external OES camera texture
    private fun ensureArSession(): Boolean {
        // Ensure camera permission
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Log.e(TAG, "ARCore: camera permission not granted")
            return false
        }
        try {
            if (arSession == null) {
                val status = ArCoreApk.getInstance().requestInstall(requireActivity(), arInstallRequested)
                if (status == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    arInstallRequested = false
                    Log.i(TAG, "ARCore: install requested; waiting for onResume")
                    return false
                }
                arSession = Session(requireActivity())
                Log.i(TAG, "ARCore: session created")
            }
            return true
        } catch (e: Exception) {
            Log.e(TAG, "ARCore: session create failed (" + e.javaClass.simpleName + "): " + (e.message ?: ""))
            return false
        }
    }

    private fun startArLoop() {
        if (!ensureArSession()) return
        if (arRunning) return
        arRunning = true
        backgroundExecutor.execute {
            if (oesConv == null) {
                oesConv = OesToRgbaConverter(targetW, targetH)
                try { arSession?.setCameraTextureNames(intArrayOf(oesConv!!.getExternalTextureId())) } catch (_: Exception) {}
            }
            // Resume on UI thread after texture id is provided
            val latch = java.util.concurrent.CountDownLatch(1)
            activity?.runOnUiThread {
                try {
                    // Configure session: camera config > 640x480, autofocus enabled, torch on
                    try {
                        val s = arSession
                        if (s != null) {
                            try {
                                val filter = com.google.ar.core.CameraConfigFilter(s)
                                    .setFacingDirection(com.google.ar.core.CameraConfig.FacingDirection.BACK)
                                val configs = s.getSupportedCameraConfigs(filter)
                                var chosen: com.google.ar.core.CameraConfig? = null
                                for (cc in configs) {
                                    val sz = cc.imageSize
                                    if (sz.width > 640 && sz.height > 480) { chosen = cc; break }
                                }
                                if (chosen == null && configs.isNotEmpty()) chosen = configs[0]
                                if (chosen != null) {
                                    s.setCameraConfig(chosen)
                                    Log.i(TAG, "ARCore: using camera config ${'$'}{chosen.imageSize.width}x${'$'}{chosen.imageSize.height}")
                                } else {
                                    Log.w(TAG, "ARCore: no suitable camera config found")
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "ARCore: camera config selection failed: ${'$'}{e.message}")
                            }

                            val cfg = com.google.ar.core.Config(s)
                            try {
                                cfg.setFocusMode(com.google.ar.core.Config.FocusMode.AUTO)
                            } catch (_: Exception) {}
                            try {
                                cfg.setFlashMode(com.google.ar.core.Config.FlashMode.TORCH)
                            } catch (_: Exception) {}
                            s.configure(cfg)
                        }
                    } catch (_: Exception) {}
                    arSession?.resume()
                    Log.i(TAG, "ARCore: session resumed")
                } catch (e: Exception) {
                    Log.e(TAG, "ARCore: resume failed (" + e.javaClass.simpleName + "): " + (e.message ?: ""))
                } finally {
                    latch.countDown()
                }
            }
            try { latch.await() } catch (_: InterruptedException) {}
            activity?.runOnUiThread {
                Toast.makeText(requireContext(), "ARCore session started", Toast.LENGTH_SHORT).show()
            }
            arUpdateLoop()
        }
    }

    private fun stopArLoop() {
        arRunning = false
        try { arSession?.pause() } catch (_: Exception) {}
        try { oesConv?.release() } catch (_: Exception) {}
        oesConv = null
    }

    private fun arUpdateLoop() {
        val session = arSession ?: return
        while (arRunning) {
            try {
                val frame: Frame = session.update()
                // AR pose is not shown to avoid overriding the gesture label

                val rgba = oesConv!!.convert(frame)
                val mpImage = ByteBufferImageBuilder(rgba, targetW, targetH, MPImage.IMAGE_FORMAT_RGBA).build()
                val frameTime = android.os.SystemClock.uptimeMillis()
                lastInputWidth = targetW
                lastInputHeight = targetH
                handLandmarkerHelper.detectAsync(mpImage, frameTime)
                gestureRecognizer?.recognizeAsync(mpImage, frameTime)
            } catch (e: Exception) {
                Log.e(TAG, "AR loop error: ${'$'}{e.message}")
            }
        }
    }
    
}
