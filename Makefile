deploy:
	docker compose --env-file .env up -d --build
deploy-d:
	docker compose --env-file .env up --build
deploy-gpu: 
	docker compose -f docker-compose-gpu.yml --env-file .env up -d --build
dev:
	docker compose -f docker-compose-dev.yml up --build
tests:
	docker compose -f docker-compose-test.yml up --build
undeploy:
	docker compose down
stop:
	docker compose stop


act-run-tests:
	gh act -j test -W '.github/workflows/tests.yml'