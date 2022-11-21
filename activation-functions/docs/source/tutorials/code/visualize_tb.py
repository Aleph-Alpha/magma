from rational.torch import Rational

Rational.list = Rational.list[:4]

for epoch in range(init_epoch, config.n_epochs + 1):
    print('- Learning:')
    # learning step
    pi.set_epsilon(epsilon)
    mdp.set_episode_end(True)
    core.learn(n_steps=config.evaluation_frequency,
               n_steps_per_fit=config.train_frequency)
    print('- Evaluation:')
    # evaluation step
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(False)
    Rational.save_all_inputs(True)
    dataset = core.evaluate(n_steps=config.test_samples)
    score = get_stats(dataset)
    writer.add_scalar(f'{args.game}/Min Reward', score[0], epoch)
    writer.add_scalar(f'{args.game}/Max Reward', score[1], epoch)
    writer.add_scalar(f'{args.game}/Mean Reward', score[2], epoch)
    Rational.show_all(writer=writer, step=epoch)
    Rational.save_all_inputs(False)
