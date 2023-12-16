deploy:
	docker compose --env-file .env up -d --build
deploy-gpu: 
	docker compose -f docker-compose-gpu.yml --env-file .env up -d --build
undeploy:
	docker compose down
stop:
	docker compose stop