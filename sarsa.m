grid_cells = 25;
actions_num = 6;
passenger_states = 5;
goal_state = 4;
grid = zeros(grid_cells,actions_num,passenger_states,goal_state);

% Initializing the grid boundaries

grid(1,1,:,:) = -1;
grid(1,3,:,:) = -1;
grid(2,1,:,:) = -1;
grid(2,4,:,:) = -1;
grid(3,1,:,:) = -1;
grid(3,3,:,:) = -1;
grid(4,1,:,:) = -1;
grid(5,1,:,:) = -1;
grid(5,4,:,:) = -1;

grid(6,3,:,:) = -1;
grid(7,4,:,:) = -1;
grid(8,3,:,:) = -1;
grid(10,4,:,:) = -1;

grid(11,3,:,:) = -1;
grid(15,4,:,:) = -1;

grid(16,3,:,:) = -1;
grid(16,4,:,:) = -1;
grid(17,3,:,:) = -1;
grid(18,4,:,:) = -1;
grid(19,3,:,:) = -1;
grid(20,4,:,:) = -1;

grid(21,2,:,:) = -1;
grid(21,3,:,:) = -1;
grid(21,4,:,:) = -1;
grid(22,2,:,:) = -1;
grid(22,3,:,:) = -1;
grid(23,2,:,:) = -1;
grid(23,4,:,:) = -1;
grid(24,2,:,:) = -1;
grid(24,3,:,:) = -1;
grid(25,2,:,:) = -1;
grid(25,4,:,:) = -1;

grid(:,5,:,:) = -1;
grid(1,5,1,:) = 1;
grid(5,5,2,:) = 1;
grid(21,5,3,:) = 1;
grid(24,5,4,:) = 1;
grid(:,5,5,:) = -1;

grid(:,6,:,:) = -1;
grid(1,6,5,1) = 10;
grid(5,6,5,2) = 10;
grid(21,6,5,3) = 10;
grid(24,6,5,4) = 10;

Q = zeros(grid_cells,actions_num,passenger_states,goal_state);  % Initializing the Q Learning Matrix
NumIter = 10000; % Number of iterations for the agent to learn
alpha = 0.1;     % Learning Factor
gamma = 0.9;     % Discount Factor
rewardArr = double(zeros(NumIter,1)); % Reward after each iteration
avgReward = double(zeros(NumIter,1)); % Average reward after each iteration
exploration_factor = 10;              % Exploration rate, also used to reduce the learning rate and to increase the discount factor

% Initialising current action to a random value.
current_action = ceil(6*rand);
    while current_action == 0
        current_action = ceil(25*rand);
    end

for i = 1:NumIter
    % Initialising the initial position of taxi in grid
    taxi_position = ceil(25*rand);
    while taxi_position == 0
        taxi_position = ceil(25*rand);
    end
    
    % Initializing the destination of passenger
    destination = ceil(4*rand);
    while destination == 0
        destination = ceil(4*rand);
    end
    
    % Assigning initial state to the passenger
    passenger = ceil(5*rand);
    while passenger == 0
        passenger = ceil(5*rand);
    end
    
    steps = 1; % Number of moves by agent in an iteration
    reward = 0.0; % Reward value in current iteration
    newpos = taxi_position;
    
    % Adjusting the learning rate, discount factor and exploration rate
    % after every 200 iterations.
    if mod(i,200) == 0
        exploration_factor = exploration_factor - 1;
        alpha = alpha - 0.01;
        gamma = gamma + 0.01;
    end
    if alpha < 0.01
        alpha = 0.01;
    end
    if gamma > 1
        gamma = 1;
    end
    
    
    while 1
        % Deciding whether to explore or play safe 
        epsilon = ceil(100*rand);
        while(epsilon == 0)
            epsilon = ceil(100*rand);
        end
        if epsilon <= exploration_factor    % Explore the grid! Do a random action.
            next_action = ceil(6*rand);
            while(next_action == 0)
                next_action = ceil(6*rand);
            end
            
        else
            [maxq,next_action] = max(Q(taxi_position,:,passenger,destination)); % Play safe! Take the most optimal action.
            if(maxq == 0)
                z = find(Q(taxi_position,:,passenger,destination));
                if isempty(z)
                    next_action = ceil(6*rand);
                    while(next_action == 0)
                        next_action = ceil(6*rand);
                    end                 
                else
                    while 1
                        next_action = ceil(6*rand);
                        while(next_action == 0)
                            next_action = ceil(6*rand);
                        end
                        if Q(taxi_position,next_action,passenger,destination) == 0
                            break;
                        end
                    end
                end
            end
        end
        
        current_reward = grid(taxi_position,current_action,passenger,destination); % Get the immediate reward value of the chosen action for the particular state
        if grid(taxi_position,current_action,passenger,destination) ~= -1   % The taxi doesn't bump into a wall.
            switch current_action
                case 1 % Action: Go North
                    if taxi_position > 5
                        newpos = taxi_position - 5;
                    end
                case 2 % Action: Go South
                    if taxi_position < 20
                        newpos = taxi_position + 5;
                    end
                case 3 % Action: Go West
                    if mod(taxi_position,5) ~= 1
                        newpos = taxi_position - 1;
                    end
                case 4 % Action: Go East
                    if mod(taxi_position,5) ~= 0
                        newpos = taxi_position + 1;
                    end
                case 5 % Action: Pick up Passenger
                    passenger = 5;
                    steps = 0;
                case 6 % Action: Drop Passenger and calculate the final reward value.
                    reward = reward + current_reward/steps;
                    rewardArr(i,1) = reward;
                    Q(taxi_position,current_action,passenger,destination) = Q(taxi_position,current_action,passenger,destination) + alpha * (grid(taxi_position,current_action,passenger,destination) + gamma*Q(newpos,next_action,passenger,destination) - Q(taxi_position,current_action,passenger,destination));
                    avgReward(i,1) = mean(rewardArr(1:i,1));
                    break;
            end
        end
        reward = reward + current_reward;
        % Abort iteration if more than 15 steps have been taken
        if steps > 15
            avgReward(i,1) = mean(rewardArr(1:i,1));
            % Drop the passenger as destination could not be found in 15 steps.
            if passenger == 5
                Q(taxi_position,5,passenger,destination) = Q(taxi_position,5,passenger,destination) + alpha * (-1 + gamma*(max(Q(newpos,:,passenger,destination))) - Q(taxi_position,5,passenger,destination));
                rewardArr(i,1) = double(reward) - 1; % Negative Reward for not dropping the passenger at the destination.
            else
                Q(taxi_position,current_action,passenger,destination) = Q(taxi_position,current_action,passenger,destination) + alpha * (grid(taxi_position,current_action,passenger,destination) + gamma*(max(Q(newpos,:,passenger,destination))) - Q(taxi_position,current_action,passenger,destination));
            end
            break;
        end
        % Update the Q value for the state
        Q(taxi_position,current_action,passenger,destination) = Q(taxi_position,current_action,passenger,destination) + alpha * (grid(taxi_position,current_action,passenger,destination) + gamma*Q(newpos,next_action,passenger,destination) - Q(taxi_position,current_action,passenger,destination));
        avgReward(i,1) = mean(rewardArr(1:i,1));
        taxi_position = newpos;
        steps = steps + 1;
        current_action = next_action;
    end
end
plot(avgReward);
