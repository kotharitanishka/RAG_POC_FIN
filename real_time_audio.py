import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play
import time

class RealtimeTranslator:
    def __init__(self, source_lang='hi', target_lang='en'):
        self.recognizer = sr.Recognizer()
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Optimize recognizer settings
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def listen_and_translate(self):
        with sr.Microphone() as source:
            print(f"\n🎤 Listening in {self.source_lang}... (Speak now)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                # Speech to Text
                print("⏳ Processing speech...")
                text = self.recognizer.recognize_google(audio, language=self.source_lang)
                print(f"✅ Original ({self.source_lang}): {text}")

                # # Translate
                # print("🔄 Translating...")
                # translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                # translated_text = translator.translate(text)
                # print(f"✅ Translated ({self.target_lang}): {translated_text}")

                # # Text to Speech
                # print("🔊 Converting to speech...")
                # tts = gTTS(text=translated_text, lang=self.target_lang, slow=False)
                # filename = f"output_{int(time.time())}.mp3"
                # tts.save(filename)

                # # Play audio
                # audio_segment = AudioSegment.from_mp3(filename)
                # play(audio_segment)

                # Cleanup
                time.sleep(0.3)
                # if os.path.exists(filename):
                #     os.remove(filename)

                return True

            except sr.WaitTimeoutError:
                print("⏰ No speech detected (timeout)")
                return False
            except sr.UnknownValueError:
                print("❌ Could not understand audio - please speak clearly")
                return False
            except Exception as e:
                print(f"❌ Error: {e}")
                return False

def main():
    print("=" * 50)
    print("Real-time Audio Translator for Indian Languages")
    print("=" * 50)

    # Language options
    print("\nSupported languages:")
    print("hi - Hindi, ta - Tamil, te - Telugu")
    print("bn - Bengali, mr - Marathi, gu - Gujarati")
    print("kn - Kannada, ml - Malayalam, en - English")

    source = input("\nEnter source language code (default: hi): ").strip() or 'hi'
    target = input("Enter target language code (default: en): ").strip() or 'en'

    translator = RealtimeTranslator(source_lang=source, target_lang=target)

    print(f"\n✨ Translator ready! ({source} → {target})")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            translator.listen_and_translate()

            print("\n" + "-" * 50)
            user_input = input("Continue? (y/n or just Enter for yes): ").strip().lower()
            if user_input and user_input != 'y':
                break
    except KeyboardInterrupt:
        print("\n\n👋 Exiting translator. Goodbye!")

if __name__ == "__main__":
    main()