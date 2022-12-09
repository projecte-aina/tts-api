speech_speed ?= 1.0

deploy:
	speech_speed=$(speech_speed) docker compose up --build
undeploy:
	docker compose down
stop:
	docker compose stop