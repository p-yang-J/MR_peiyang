using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Input;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;

public class interaction_data2 : MonoBehaviour
{
    public string serverIp = "10.167.156.160";
    public int serverPort = 16306;

    private UdpClient udpClient;

    void Start()
    {
        udpClient = new UdpClient();
    }

    void Update()
    {
        foreach (var controller in CoreServices.InputSystem.DetectedControllers)
        {
            if (controller.ControllerHandedness == Handedness.Right &&
                controller is IMixedRealityHand handController)
            {
                if (handController.TryGetJoint(TrackedHandJoint.IndexTip, out MixedRealityPose pose))
                {
                    Vector3 handPosition = pose.Position;
                    string handPositionData = $"HandPosition: {handPosition.x}, {handPosition.y}, {handPosition.z}";
                    byte[] handPositionBytes = Encoding.UTF8.GetBytes(handPositionData);
                    udpClient.Send(handPositionBytes, handPositionBytes.Length, serverIp, serverPort);
                    transform.position = new Vector3(handPosition.x, handPosition.y, handPosition.z);
                }
            }
        }

        if (CoreServices.InputSystem.EyeGazeProvider.IsEyeTrackingEnabledAndValid)
        {
            var eyeGazeOrigin = CoreServices.InputSystem.EyeGazeProvider.GazeOrigin;
            var eyeGazeDirection = CoreServices.InputSystem.EyeGazeProvider.GazeDirection;
            string gazeData = $"GazeOrigin: {eyeGazeOrigin.x}, {eyeGazeOrigin.y}, {eyeGazeOrigin.z}, GazeDirection: {eyeGazeDirection.x}, {eyeGazeDirection.y}, {eyeGazeDirection.z}";
            byte[] gazeBytes = Encoding.UTF8.GetBytes(gazeData);
            udpClient.Send(gazeBytes, gazeBytes.Length, serverIp, serverPort);
        }
    }

    private void OnDestroy()
    {
        udpClient.Close();
    }
}
