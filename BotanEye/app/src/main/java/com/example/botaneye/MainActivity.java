package com.example.botaneye;

import static android.content.ContentValues.TAG;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.squareup.picasso.Picasso;
//import com.theartofdev.edmodo.cropper.CropImage;



import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    byte[] by;
    String username;
    private String encodedImage="";
    ImageView userpic;
    Button b1;
    private static final int GalleryPick = 1;
    private static final int CAMERA_REQUEST = 100;
    private static final int STORAGE_REQUEST = 200;
    private static final int IMAGEPICK_GALLERY_REQUEST = 300;
    private static final int IMAGE_PICKCAMERA_REQUEST = 400;
    String cameraPermission[];
    String storagePermission[];
    Uri imageuri;

    TextView click;
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        click = findViewById(R.id.click);
        userpic = findViewById(R.id.set_profile_image);
        b1=findViewById(R.id.bid);

        // allowing permissions of gallery and camera
        cameraPermission = new String[]{android.Manifest.permission.CAMERA, android.Manifest.permission.WRITE_EXTERNAL_STORAGE};
        storagePermission = new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE};

        // After clicking on text we will have
        // to choose whether to
        // select image from camera and gallery
        click.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                showImagePicDialog();

            }
        });

        b1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (encodedImage.equals("")){
                    Toast.makeText(MainActivity.this,"Please select an image",Toast.LENGTH_LONG).show();
                }
                else {


                    Log.e("Encoded string==>",encodedImage);
//                    Toast.makeText(MainActivity.this,"Please Wait, Your image is processing!!",Toast.LENGTH_LONG).show();
                    Toast.makeText(MainActivity.this,"Please Wait, Your image is processing!!",Toast.LENGTH_LONG).show();
                    RequestQueue requestQueue= Volley.newRequestQueue(getApplicationContext());
                    StringRequest requ=new StringRequest(Request.Method.POST, "http://192.168.161.137:8000/Upload_image/", new Response.Listener<String>() {
                        @Override
                        public void onResponse(String response) {

                            Log.e("Response is: ", response.toString());
                            try {
                                JSONObject o = new JSONObject(response);
                                String dat = o.getString("msg");
                                String desc = o.getString("def");
                                if(dat.equals("no")){
                                    Toast.makeText(MainActivity.this, "Some error occured!", Toast.LENGTH_LONG).show();
//                                    Intent i1=new Intent(Upload_item.this,Userhome.class);
//                                    startActivity(i1);

                                }
                                else {
                                    Toast.makeText(MainActivity.this, "Process completed", Toast.LENGTH_LONG).show();
                                    Intent i1=new Intent(MainActivity.this,View_result.class);
                                    i1.putExtra("result",dat);
                                    i1.putExtra("desc",desc);
                                    startActivity(i1);
                                }
                            }
                            catch (Exception e){
                                e.printStackTrace();

                            }

                        }
                    }, new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
//                Log.e(TAG,error.getMessage());
                            error.printStackTrace();
                        }
                    }){
                        @Override
                        protected Map<String, String> getParams() throws AuthFailureError {
                            Map<String,String> m=new HashMap<>();
                            m.put("encodedstr",encodedImage);



                            return m;
                        }
                    };
                    requ.setRetryPolicy(new DefaultRetryPolicy(
                            90000,  // Timeout in milliseconds
                            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
                    ));
                    requestQueue.add(requ);



                }
            }
        });

    }
    private void showImagePicDialog() {
        String options[] = {"Gallery"};//{"Camera", "Gallery"}
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Pick Image From");
        builder.setItems(options, new DialogInterface.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (which == 0) {
                    if (!checkCameraPermission()) {
                        requestCameraPermission();
                    } else {
                        pickFromGallery();
                    }
                } else if (which == 1) {
                    if (!checkStoragePermission()) {
                        requestStoragePermission();
                    } else {
                        pickFromGallery();
                    }
                }
            }
        });
        builder.create().show();
    }

    // checking storage permissions
    private Boolean checkStoragePermission() {
        boolean result = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == (PackageManager.PERMISSION_GRANTED);
        return result;
    }

    // Requesting gallery permission
    @RequiresApi(api = Build.VERSION_CODES.M)
    private void requestStoragePermission() {
        requestPermissions(storagePermission, STORAGE_REQUEST);
    }

    // checking camera permissions
    private Boolean checkCameraPermission() {
        boolean result = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == (PackageManager.PERMISSION_GRANTED);
        boolean result1 = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == (PackageManager.PERMISSION_GRANTED);
        return result && result1;
    }

    // Requesting camera permission
    @RequiresApi(api = Build.VERSION_CODES.M)
    private void requestCameraPermission() {
        requestPermissions(cameraPermission, CAMERA_REQUEST);
    }

    // Requesting camera and gallery
    // permission if not given
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case CAMERA_REQUEST: {
                if (grantResults.length > 0) {
                    boolean camera_accepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean writeStorageaccepted = grantResults[1] == PackageManager.PERMISSION_GRANTED;
                    if (camera_accepted && writeStorageaccepted) {
                        pickFromGallery();
                    } else {
                        Toast.makeText(this, "Please Enable Camera and Storage Permissions", Toast.LENGTH_LONG).show();
                    }
                }
            }
            break;
            case STORAGE_REQUEST: {
                if (grantResults.length > 0) {
                    boolean writeStorageaccepted = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    if (writeStorageaccepted) {
                        pickFromGallery();
                    } else {
                        Toast.makeText(this, "Please Enable Storage Permissions", Toast.LENGTH_LONG).show();
                    }
                }
            }
            break;
        }
    }
    private void pickFromGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK);
        galleryIntent.setType("image/*");
        startActivityForResult(galleryIntent, IMAGEPICK_GALLERY_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == IMAGEPICK_GALLERY_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();

            Log.e("Image URI-->", imageUri + "");
            Picasso.with(this).load(imageUri).into(userpic);

            try {
                Bitmap photo = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                photo.compress(Bitmap.CompressFormat.JPEG, 100, baos);
                by = baos.toByteArray();
                encodedImage = Base64.encodeToString(by, Base64.DEFAULT);
                Log.e(TAG, encodedImage);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}