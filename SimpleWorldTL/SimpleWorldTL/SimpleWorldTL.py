from pynput import keyboard

def on_press(key):
    try:
        print(f'Key {key} pressed')
    except AttributeError:
        print(' {0} pressed'.format(key))

def on_release(key):
    print(f'Key {key} released')
    if key == keyboard.Key.esc:
        # Esc 키를 누르면 리스너를 중지
        return False

# 키보드 리스너 시작
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()