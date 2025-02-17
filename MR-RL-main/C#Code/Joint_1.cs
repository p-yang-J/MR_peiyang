using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;

public class Joint_1 : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 16308;
    public Vector3 receivedPos = Vector3.zero;

    private void Start()
    {
        //Debug.Log("Script started");
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void Update()
    {
        // Debug.Log("Cube position: " + receivedPos);
        float x = receivedPos.x;
        transform.position = new Vector3(0.046f, 0.09855201f, x);
    }



    private void OnDestroy()
    {
        if (receiveThread != null)
            receiveThread.Abort();

        client.Close();
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);

                string text = Encoding.UTF8.GetString(data);
                //Debug.Log("Received raw data: " + text);
                receivedPos = StringToVector3(text);
                //Debug.Log("Parsed received data: " + receivedPos);
            }
            catch (SocketException e)
            {
                //Debug.Log("SocketException: " + e);
            }
        }
    }



    public static Vector3 StringToVector3(string sVector)
    {
        if (sVector.StartsWith("(") && sVector.EndsWith(")"))
        {
            sVector = sVector.Substring(1, sVector.Length - 2);
        }

        string[] sArray = sVector.Split(',');
        Vector3 result = new Vector3(
            float.Parse(sArray[0]),
            float.Parse(sArray[1]),
            float.Parse(sArray[2]));

        return result;
    }
}
