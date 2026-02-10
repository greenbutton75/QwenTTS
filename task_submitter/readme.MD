#Описание работы 

## Создание профиля
Имеет AWS креды для записи mp3 образцов голоса в бакет rixtrema-qwentts/support/{user_id}/voices/{voice_id}/
Загружает mp3 (или любой поддерживаемый формат) образца голоса в бакет
Затем вызывает метод 
    task_type = "QWEN_TTS_CREATE_PROFILE"
    file_id = None
    task_id = create_task(task_type, file_id,{"support_id": "user1_46847", "voice_id": "voice_21327ef670", "voice_name": "Name of Voice 2", "ref_text" : "Each book in the series was originally published in hardcover format with a number of full-color illustrations spread throughout."})

Таска помещается в очередь и сервис на машине с GPU её подберет.

## Озвучивание фразы

вызывает метод 
    task_type = "QWEN_TTS_PHRASE"
    file_id = None
    task_id = create_task(task_type, file_id,{"support_id": "user1_46847", "voice_id": "voice_21327ef670", "phrase_id" : "phrase_006", "text" : "Protect and distribute your assets according to your wishes with our comprehensive trust administration services."})



#Конкретные вызовы

##Создание профиля голоса

python -m task_submitter.main create-profile `
  --support-id user1_46847 `
  --voice-id voice_2 `
  --voice-name "Sveta sample" `
  --ref-text "Each book in the series was originally published in hardcover format with a number of full-color illustrations spread throughout." `
  --sample "D:\Work\ML\Voice\Sveta.m4a"

##Создание фразы
python -m task_submitter.main create-phrase `
  --support-id user1_46847 `
  --voice-id voice_2 `
  --phrase-id phrase_011 `
  --text "We used SPY because we already maintain a reliable daily price history for it internally, which allowed us to implement this change consistently across all plans."


Результаты в таблице select top 90 * from TasksManager..Tasks  (ну и в очереди задач тоже)

Id	FileId	Created	Type	TaskParameters	Completed	Status	Progress	Stage	Data
118480	NULL	20260210122609	QWEN_TTS_CREATE_PROFILE	{"support_id": "user1_46847", "voice_id": "voice_2", "voice_name": "Sveta sample", "ref_text": "Each book in the series was originally published in hardcover format with a number of full-color illustrations spread throughout."}	20260210122837	SUCCESS	100	COMPLETED	done
118483	NULL	20260210122905	QWEN_TTS_PHRASE	{"support_id": "user1_46847", "voice_id": "voice_2", "phrase_id": "phrase_011", "text": "We used SPY because we already maintain a reliable daily price history for it internally, which allowed us to implement this change consistently across all plans."}	20260210123011	SUCCESS	100	COMPLETED	https://rixtrema-qwentts.s3.amazonaws.com/support/user1_46847/phrases/phrase_011.wav?AWSAccessKeyId=AKIA322GJ6UJFW6EAGOZ&Signature=pCA1zxu8f0NGD5z4IcMRYWQKOgs%3D&Expires=1775910583
118484	NULL	20260210123012	QWEN_TTS_READY_PHRASE	{"support_id": "user1_46847", "voice_id": "voice_2", "phrase_id": "phrase_011"}	NULL	NEW	0	START	NULL
118482	NULL	20260210122838	QWEN_TTS_READY_PROFILE	{"support_id": "user1_46847", "voice_id": "voice_2"}	NULL	NEW	0	START	NULL



#Мониторинг
Эти IP и порты - для текущего запуска инстанса.
http://24.124.32.70:57990/health (API health)
http://24.124.32.70:57990/admin
http://24.124.32.70:57643/health (task_worker health)

http://admin:jzexxxxxxxxxxxxxxx@24.124.32.70:57990/admin
