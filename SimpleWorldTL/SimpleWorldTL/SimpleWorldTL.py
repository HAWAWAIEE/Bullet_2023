from pynput import keyboard

def on_press(key):
    try:
        print(f'Key {key} pressed')
    except AttributeError:
        print(' {0} pressed'.format(key))

def on_release(key):
    print(f'Key {key} released')
    if key == keyboard.Key.esc:
        # Esc Ű�� ������ �����ʸ� ����
        return False

# Ű���� ������ ����
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()