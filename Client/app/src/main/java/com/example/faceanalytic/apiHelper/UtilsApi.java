package com.example.faceanalytic.apiHelper;

public class UtilsApi {
    public static final String BASE_URL_API = "https://bangkit-face-exp.df.r.appspot.com/";

    public static BaseApiService getAPIService(){
        return RetrofitClient.getClient(BASE_URL_API).create(BaseApiService.class);
    }
}
