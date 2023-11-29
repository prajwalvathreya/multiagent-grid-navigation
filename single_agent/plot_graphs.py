import matplotlib.pyplot as plt
import json
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

# Specify the font
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

# Load data from rewards.json
with open("rewards.json", 'r') as json_file:
    rewards = json.load(json_file)

# Load data from loss.json
with open("loss.json", 'r') as json_file:
    losses = json.load(json_file)

# Get the first entry for plotting
plot_reward = next(iter(rewards.values()), None)
plot_loss = next(iter(losses.values()), None)

# Plotting
episodes = list(range(1, len(plot_loss) + 1))  # Assuming the length of plot_loss is the number of episodes

# Beautify the plot
def beautify_plot(ax, type):
    ax.set_xlabel('Episode', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel(type, fontsize=12, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Show integer ticks on the x-axis

# Plot Loss
plt.figure(figsize=(8, 6))
plt.fill_between(episodes, plot_loss, color='#FF6800', alpha=0.6, label='Loss Area')
plt.title('Episode vs Loss', fontsize=14, fontname='Times New Roman')
beautify_plot(plt.gca(), 'Loss')  # Get the current axes
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('newloss_plot.jpg')
plt.clf()

# Plot Reward
plt.figure(figsize=(8, 6))
plt.fill_between(episodes, plot_reward, color='skyblue', alpha=0.8, label='Reward Area')
plt.title('Episode vs Reward', fontsize=14, fontname='Times New Roman')
beautify_plot(plt.gca(), 'Reward')  # Get the current axes
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('newrewards_plot.jpg')
plt.clf()


### The below code is to print the average reward across all the rewards and losses

# Calculate average rewards and losses
overall_rewards = [sum(values) / len(values) for values in zip(*rewards.values())]
overall_losses = [sum(values) / len(values) for values in zip(*losses.values())]


environments = 250

# Plot Loss
plt.figure(figsize=(8, 6))
plt.fill_between(environments, plot_loss, color='#FF6800', alpha=0.6, label='Loss Area')
plt.title('Environment vs Avg Loss', fontsize=14, fontname='Times New Roman')
beautify_plot(plt.gca(), 'Avg Loss')  # Get the current axes
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_loss_plot.jpg')
plt.clf()

# Plot Reward
plt.figure(figsize=(8, 6))
plt.fill_between(environments, plot_reward, color='skyblue', alpha=0.8, label='Reward Area')
plt.title('Environment vs Avg Reward', fontsize=14, fontname='Times New Roman')
beautify_plot(plt.gca(), 'Avg Reward')  # Get the current axes
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_rewards_plot.jpg')
plt.clf()
