speech_speed ?= 1.0
mp_workers ?= 4

deploy:
	speech_speed=$(speech_speed) mp_workers=$(mp_workers) docker compose up --build
undeploy:
	docker compose down
stop:
	docker compose stop