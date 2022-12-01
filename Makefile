speed ?= 1.0

deploy:
	speed=$(speed) docker compose up --build
undeploy:
	docker compose down
stop:
	docker compose stop