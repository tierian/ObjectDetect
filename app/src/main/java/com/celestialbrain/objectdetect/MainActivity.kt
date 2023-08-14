package com.celestialbrain.objectdetect

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.FileUtils
import android.provider.MediaStore
import android.provider.Settings
import android.widget.Button
import android.widget.ImageView
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import com.celestialbrain.objectdetect.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    val paint = Paint()
    lateinit var imageView:ImageView
    lateinit var button:Button
    lateinit var bitmap: Bitmap
    lateinit var model:SsdMobilenetV11Metadata1
    lateinit var labels:List<String>
    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300,300, ResizeOp.ResizeMethod.BILINEAR)).build()


    val selectImgResult =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result: ActivityResult? ->
            var uri = result?.data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            get_predictions()
        }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

        labels = FileUtil.loadLabels(this,"label.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this)

        imageView = findViewById<ImageView>(R.id.imageView)
        button = findViewById<Button>(R.id.button)


        button.setOnClickListener{
            selectImgResult.launch(intent)
        }
    }



    fun get_predictions(){


        // Creates inputs for reference.
        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)

        // Runs model inference and gets result.
        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray
        val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

        var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)

        val h = mutable.height
        val w = mutable.width
        var x = 0

        //文字と枠の太さ
        paint.textSize = h/15f //数字が少ないほど大きくなる
        paint.strokeWidth = h/85f


        scores.forEachIndexed{index, fl ->
            x = index
            if(fl > 0.5){
                paint.style = Paint.Style.STROKE
                canvas.drawRect(RectF(locations.get(x+1)*w,locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h),paint)
                paint.style = Paint.Style.FILL
                canvas.drawText(labels.get(classes.get(index).toInt())+" "+fl.toString(),locations.get(x+1)*w,locations.get(x)*h,paint)
            }
        }

        imageView.setImageBitmap(mutable)
    }

    override fun onDestroy() {
        super.onDestroy()

        // Releases model resources if no longer used.
        model.close()
    }
}