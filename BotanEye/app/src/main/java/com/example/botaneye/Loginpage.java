package com.example.botaneye;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class Loginpage extends AppCompatActivity {
    EditText ed1,ed2;
    Button b1;
    TextView tv1;
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_loginpage);

//        ActionBar actionBar = getSupportActionBar();
//        actionBar.hide();
        ed1=(EditText)findViewById(R.id.unm);
        ed2=(EditText)findViewById(R.id.pass);
        b1=(Button) findViewById(R.id.button);

        b1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String uname=ed1.getText().toString();
                String pass=ed2.getText().toString();
                Log.e("uname pass-->",uname+" "+pass);

                if (uname.equals("")||pass.equals("")){
                    Toast.makeText(getApplicationContext(),"Please fill all the details",Toast.LENGTH_SHORT).show();
                }

                else {


                    RequestQueue requestQueue= Volley.newRequestQueue(getApplicationContext());
                    StringRequest requ=new StringRequest(Request.Method.POST, "http://192.168.161.137:8000/Login_check/", new Response.Listener<String>() {
                        @Override
                        public void onResponse(String response) {

                            Log.e("Response is: ", response.toString());
                            try {
                                JSONObject o = new JSONObject(response);
                                String dat = o.getString("msg");
                                if(dat.equals("yes")){
                                    Toast.makeText(Loginpage.this, "Login Successful!", Toast.LENGTH_LONG).show();
                                    Intent i1=new Intent(Loginpage.this,MainActivity.class);
                                    startActivity(i1);
                                }
                                else if(dat.equals("invalid"))
                                {
                                    Toast.makeText(Loginpage.this, "Invalid Credentials!", Toast.LENGTH_LONG).show();
                                }
                                else {
                                    Toast.makeText(Loginpage.this, "No account found!", Toast.LENGTH_LONG).show();
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
                            m.put("uname",uname);
                            m.put("pswrd",pass);


                            return m;
                        }
                    };
                    requestQueue.add(requ);
                }
            }
        });
    }
}