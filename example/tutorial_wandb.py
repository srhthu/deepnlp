import wandb

# modify config of finished runs
api = wandb.api()
r = api.run('next-nlp/legal-tuning/klda0w6c')
print(r.group)
print(r.tags)
# do some modification
r.update()