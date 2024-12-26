# from django.shortcuts import render

# # def index(request):
# #     return render(request, 'translator/index.html')

# # # def translate_text(request):
# # #     if request.method == "POST":
# # #         input_text = request.POST.get("input_text", "")
# # #         if input_text:
# # #             translation = translate(input_text)
# # #             return render(request, 'translator/result.html', {
# # #                 "input_text": input_text,
# # #                 "translation": translation,
# # #             })
# # #     return render(request, 'translator/result.html', {"error": "Введите текст для перевода."})

# # from django.shortcuts import render

# # # Функция для перевода текста

# # # Обработчик для перевода текста
# # def translate_text(request):
# #     """
# #     Обработчик для перевода текста.
# #     """
# #     print("[DEBUG] Entering translate_text function...")

# #     if request.method == "POST":
# #         print("[DEBUG] POST request detected.")

# #         # Получаем введённый текст из формы
# #         input_text = request.POST.get("input_text", "").strip()
# #         print(f"[DEBUG] Received input_text: '{input_text}'")

# #         if not input_text:
# #             print("[DEBUG] No input_text provided.")
# #             # Если текст не введён, показываем сообщение об ошибке
# #             return render(request, "translator/index.html", {"error": "Введите текст для перевода."})

# #         try:
# #             print("[DEBUG] Calling translate function...")

# #             # Выполняем перевод текста
# #             translation = translate(input_text)

# #             print(f"[DEBUG] Translation result: '{translation}'")

# #             # Возвращаем результат перевода
# #             return render(request, "translator/result.html", {
# #                 "input_text": input_text,
# #                 "translation": translation
# #             })
# #         except Exception as e:
# #             print(f"[ERROR] Error during translation: {e}")
# #             # Если произошла ошибка при переводе, возвращаем сообщение об ошибке
# #             return render(request, "translator/index.html", {
# #                 "error": f"Ошибка перевода: {e}"
# #             })

# #     print("[DEBUG] Request method is not POST. Returning empty form.")
# #     # Если запрос не POST, возвращаем пустую форму
# #     return render(request, "translator/index.html")

# # def translate_view(request):
# #     word = request.GET.get("word", "")
# #     result = translate(word)
# #     return JsonResponse({"input": word, "output": result})
# from django.shortcuts import render
# from .load_model import translate

# def translate_text(request):
#     """
#     Handles text translation from input form and displays the result.
#     """
#     print("[DEBUG] Entering translate_text function...")

#     if request.method == "POST":
#         print("[DEBUG] POST request detected.")

#         # Retrieve input text from the form
#         input_text = request.POST.get("input_text", "").strip()
#         print(f"[DEBUG] Received input_text: '{input_text}'")

#         if not input_text:
#             print("[DEBUG] No input_text provided.")
#             # If no text is entered, return to the input form with an error message
#             return render(request, "translator/index.html", {"error": "Введите текст для перевода."})

#         try:
#             print("[DEBUG] Calling translate function...")

#             # Perform translation
#             translation = translate(input_text)

#             print(f"[DEBUG] Translation result: '{translation}'")

#             # Render the result page with input and translation
#             return render(request, "translator/result.html", {
#                 "input_text": input_text,
#                 "translation": translation
#             })
#         except Exception as e:
#             print(f"[ERROR] Error during translation: {e}")
#             # If an error occurs, show the error message on the input form
#             return render(request, "translator/index.html", {
#                 "error": f"Ошибка перевода: {e}"
#             })

#     print("[DEBUG] Request method is not POST. Returning empty form.")
#     # If the request method is not POST, return the input form
#     return render(request, "translator/index.html")

# from django.http import JsonResponse
# from .load_model import translate

# def translate_view(request):
#     word = request.POST.get("word", "").strip()
#     if not word:
#         return JsonResponse({"error": "No word provided for translation"}, status=400)

#     try:
#         result = translate(word)
#         return JsonResponse({"input": word, "output": result})
#     except Exception as e:
#         return JsonResponse({"error": f"Translation failed: {str(e)}"}, status=500)

from django.shortcuts import render
from django.http import JsonResponse
from .load_model import translate  # Import the translation function from load_model

def translate_text(request):
    """
    Handles text translation from the input form and displays the result.
    """
    print("[DEBUG] Entering translate_text function...")

    if request.method == "POST":
        print("[DEBUG] POST request detected.")

        # Retrieve input text from the form
        input_text = request.POST.get("input_text", "").strip()
        print(f"[DEBUG] Received input_text: '{input_text}'")

        if not input_text:
            print("[DEBUG] No input_text provided.")
            # If no text is entered, return to the input form with an error message
            return render(request, "translator/index.html", {"error": "Введите текст для перевода."})

        try:
            print("[DEBUG] Calling translate function...")

            # Perform translation
            translation = translate(input_text)

            print(f"[DEBUG] Translation result: '{translation}'")

            # Render the result page with input and translation
            return render(request, "translator/result.html", {
                "input_text": input_text,
                "translation": translation
            })
        except Exception as e:
            print(f"[ERROR] Error during translation: {e}")
            # If an error occurs, show the error message on the input form
            return render(request, "translator/index.html", {
                "error": f"Ошибка перевода: {e}"
            })

    print("[DEBUG] Request method is not POST. Returning empty form.")
    # If the request method is not POST, return the input form
    return render(request, "translator/index.html")

def translate_view(request):
    """
    API endpoint for translating a word. Returns a JSON response.
    """
    print("[DEBUG] Entering translate_view function...")

    word = request.POST.get("word", "").strip()
    if not word:
        print("[DEBUG] No word provided.")
        return JsonResponse({"error": "No word provided for translation"}, status=400)

    try:
        print("[DEBUG] Calling translate function...")
        result = translate(word)
        print(f"[DEBUG] Translation result: '{result}'")
        return JsonResponse({"input": word, "output": result})
    except Exception as e:
        print(f"[ERROR] Translation failed: {str(e)}")
        return JsonResponse({"error": f"Translation failed: {str(e)}"}, status=500)
