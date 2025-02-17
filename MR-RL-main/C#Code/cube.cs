using System;
using UnityEngine;
using NativeWebSocket;

public class video : MonoBehaviour
{
    WebSocket websocket;

    async void Start()
    {
        websocket = new WebSocket("ws://localhost:8765");

        websocket.OnOpen += () =>
        {
            Debug.Log("Connection open!");
        };

        websocket.OnError += (e) =>
        {
            Debug.Log("Error! " + e);
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log("Connection closed!");
        };

        websocket.OnMessage += (bytes) =>
        {
            // 将base64字符串解码为字节数组
            byte[] jpegData = Convert.FromBase64String(System.Text.Encoding.UTF8.GetString(bytes));
            Debug.Log("Connection open1");
            Texture2D tex = new Texture2D(2, 2);
            tex.LoadImage(jpegData);  // 将JPEG数据加载到纹理
            tex.Apply();
            Debug.Log("Connection open2");
            GetComponent<Renderer>().material.mainTexture = tex;  // 将纹理应用到材质
        };

        // waiting for messages
        await websocket.Connect();
    }

    async void OnApplicationQuit()
    {
        await websocket.Close();
    }
}
