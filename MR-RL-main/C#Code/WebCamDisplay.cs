using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WebCamDisplay : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        // Create a WebCamTexture
        WebCamTexture webCamTexture = new WebCamTexture();

        // Assign the texture to the renderer's material
        GetComponent<Renderer>().material.mainTexture = webCamTexture;

        // Start the WebCamTexture
        webCamTexture.Play();
    }
}

