import cv2
import time

def test_hd60s_devices():
    """Test different device IDs to find available HD60 S+"""
    print("🔍 Testing HD60 S+ device IDs...")
    
    for device_id in range(0, 5):
        print(f"\n📹 Testing device ID: {device_id}")
        
        # Try DirectShow backend (Windows)
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # Test frame capture
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"✅ Device {device_id} - Resolution: {width}x{height}")
                
                # Check if it's actually capturing (not grey)
                mean_color = frame.mean()
                if mean_color > 10 and mean_color < 245:  # Not completely black or white
                    print(f"✅ Device {device_id} - Active video detected (mean: {mean_color:.1f})")
                    
                    # Show preview for 2 seconds
                    cv2.imshow(f"Device {device_id} Preview", frame)
                    cv2.waitKey(2000)
                    cv2.destroyAllWindows()
                    
                    cap.release()
                    return device_id
                else:
                    print(f"⚠️  Device {device_id} - Grey/blank screen (mean: {mean_color:.1f})")
            else:
                print(f"❌ Device {device_id} - Cannot capture frames")
        else:
            print(f"❌ Device {device_id} - Cannot open")
        
        cap.release()
        time.sleep(0.5)
    
    print("\n❌ No working HD60 S+ found!")
    return None

def test_nintendo_switch_capture():
    """Test HD60 S+ specifically for Nintendo Switch"""
    print("\n🎮 Nintendo Switch HD60 S+ Configuration Test")
    
    # Find working device
    device_id = test_hd60s_devices()
    if device_id is None:
        print("❌ No HD60 S+ device found. Solutions:")
        print("   1. Close Elgato Game Capture software completely")
        print("   2. Disconnect and reconnect HD60 S+")
        print("   3. Try different USB port")
        print("   4. Restart computer")
        return
    
    print(f"\n✅ Using HD60 S+ on device ID: {device_id}")
    
    # Setup for Nintendo Switch (1080p)
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    
    # Nintendo Switch optimal settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency
    
    # Try MJPG codec for better USB performance
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        print("✅ MJPG codec enabled")
    except:
        print("⚠️  Using default codec")
    
    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Final settings: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Live preview
    print("🎮 Live Nintendo Switch preview (Press 'q' to quit, 's' to save frame)")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Lost connection to HD60 S+")
            break
        
        frame_count += 1
        
        # Add info overlay
        cv2.putText(frame, f"Nintendo Switch - HD60 S+", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Device ID: {device_id}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {actual_width}x{actual_height}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Nintendo Switch - HD60 S+ Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"switch_capture_{frame_count}.jpg", frame)
            print(f"💾 Saved frame: switch_capture_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ HD60 S+ test complete!")
    print(f"📋 Use device ID {device_id} in your main script")

if __name__ == "__main__":
    test_nintendo_switch_capture()