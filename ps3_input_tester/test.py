import cv2
import time
import numpy as np

def list_available_cameras():
    """List all available camera devices"""
    print("🔍 Scanning for video capture devices...")
    available_devices = []
    
    # Test device IDs 0-10
    for device_id in range(11):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                available_devices.append({
                    'id': device_id,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'frame_shape': frame.shape
                })
                print(f"✅ Device {device_id}: {width}x{height} @ {fps}fps")
            cap.release()
        else:
            print(f"❌ Device {device_id}: Not available")
    
    return available_devices

def test_device_with_backends(device_id):
    """Test a specific device with different OpenCV backends"""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_V4L2, "V4L2 (Linux)"),
        (cv2.CAP_ANY, "Auto-detect"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
    ]
    
    print(f"\n🧪 Testing device {device_id} with different backends:")
    
    for backend_code, backend_name in backends:
        try:
            print(f"   Trying {backend_name}...")
            cap = cv2.VideoCapture(device_id, backend_code)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"   ✅ {backend_name}: {width}x{height} @ {fps}fps")
                    cap.release()
                    return backend_code, backend_name
                else:
                    print(f"   ⚠ {backend_name}: Opened but no frame captured")
                cap.release()
            else:
                print(f"   ❌ {backend_name}: Could not open")
        except Exception as e:
            print(f"   ❌ {backend_name}: Error - {e}")
    
    return None, None

def test_stream_link_capture():
    """Test Stream Link 4K capture with various settings"""
    print("\n" + "="*60)
    print("🎮 STREAM LINK 4K CAPTURE TEST")
    print("="*60)
    
    # First, list all available devices
    devices = list_available_cameras()
    
    if not devices:
        print("\n❌ No video capture devices found!")
        print("💡 Make sure your Stream Link 4K is:")
        print("   1. Connected via USB")
        print("   2. Getting power (LED indicator)")
        print("   3. Has HDMI input connected (PS3)")
        print("   4. Not being used by other software")
        return
    
    print(f"\n✅ Found {len(devices)} video device(s)")
    
    # Test each device
    for device_info in devices:
        device_id = device_info['id']
        print(f"\n" + "="*40)
        print(f"Testing Device {device_id}")
        print("="*40)
        
        # Find best backend for this device
        backend, backend_name = test_device_with_backends(device_id)
        
        if backend is None:
            print(f"⚠ Device {device_id}: No working backend found")
            continue
        
        # Try to capture with the best backend
        try:
            cap = cv2.VideoCapture(device_id, backend)
            
            if not cap.isOpened():
                print(f"❌ Could not open device {device_id}")
                continue
            
            # Try different resolutions for Stream Link 4K
            resolutions_to_try = [
                (1920, 1080),  # 1080p
                (1280, 720),   # 720p
                (640, 480),    # 480p
                (3840, 2160),  # 4K (if supported)
            ]
            
            print(f"\n🔧 Testing resolutions on device {device_id}:")
            
            working_resolution = None
            for width, height in resolutions_to_try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✅ {width}x{height} -> {actual_width}x{actual_height}")
                    if working_resolution is None:
                        working_resolution = (actual_width, actual_height)
                else:
                    print(f"   ❌ {width}x{height} -> No frame")
            
            if working_resolution:
                print(f"\n🎬 Starting live preview for device {device_id}")
                print(f"   Backend: {backend_name}")
                print(f"   Resolution: {working_resolution[0]}x{working_resolution[1]}")
                print("   Press 'q' to quit, 'n' for next device")
                
                # Live preview
                frame_count = 0
                start_time = time.time()
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        print("❌ Failed to capture frame")
                        break
                    
                    frame_count += 1
                    
                    # Add info overlay
                    info_text = f"Device {device_id} | {backend_name} | Frame {frame_count}"
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Calculate and display FPS
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        fps = frame_count / elapsed
                        fps_text = f"FPS: {fps:.1f}"
                        cv2.putText(frame, fps_text, (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Display frame info
                    height, width = frame.shape[:2]
                    res_text = f"Resolution: {width}x{height}"
                    cv2.putText(frame, res_text, (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Check if frame is all black (common issue)
                    if np.mean(frame) < 1:  # Very dark frame
                        cv2.putText(frame, "WARNING: Very dark/black frame!", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "Check HDMI input connection", (10, 190), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow(f"Stream Link 4K Test - Device {device_id}", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("👤 User quit")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('n'):
                        print("➡ Moving to next device")
                        break
                
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Error testing device {device_id}: {e}")
    
    print("\n✅ Testing complete!")

def detailed_device_info(device_id):
    """Get detailed information about a specific device"""
    print(f"\n📋 Detailed info for device {device_id}:")
    
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"❌ Cannot open device {device_id}")
        return
    
    properties = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Frame Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Frame Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_FORMAT, "Format"),
        (cv2.CAP_PROP_FOURCC, "FourCC"),
        (cv2.CAP_PROP_BUFFERSIZE, "Buffer Size"),
        (cv2.CAP_PROP_AUTO_EXPOSURE, "Auto Exposure"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_AUTOFOCUS, "Autofocus"),
    ]
    
    for prop, name in properties:
        try:
            value = cap.get(prop)
            print(f"   {name}: {value}")
        except:
            print(f"   {name}: Not supported")
    
    cap.release()

if __name__ == "__main__":
    print("🚀 Stream Link 4K Capture Device Tester")
    print("This script will help you find and test your capture device")
    
    # Run the main test
    test_stream_link_capture()
    
    # Ask user if they want detailed info on a specific device
    try:
        device_input = input("\nEnter device ID for detailed info (or press Enter to skip): ").strip()
        if device_input.isdigit():
            detailed_device_info(int(device_input))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except:
        print("\n✅ Test complete!")