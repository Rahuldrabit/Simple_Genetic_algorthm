#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

struct Individual {
    std::vector<double> genes;
    double fitness;
};

// Function implementations

double rastrigin(const std::vector<double>& x) {
    const double A = 10.0;
    double sum = A * x.size();
    for (double xi : x) {
        sum += xi * xi - A * std::cos(2 * M_PI * xi);
    }
    return sum;
}

double ackley(const std::vector<double>& x) {
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2 * M_PI;
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (double xi : x) {
        sum1 += xi * xi;
        sum2 += std::cos(c * xi);
    }
    double term1 = -a * std::exp(-b * std::sqrt(sum1 / x.size()));
    double term2 = -std::exp(sum2 / x.size());
    return term1 + term2 + a + std::exp(1.0);
}

double schwefel(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * std::sin(std::sqrt(std::fabs(xi)));
    }
    return 418.9829 * x.size() - sum;
}

using FitnessFunc = double(*)(const std::vector<double>&);

struct GAConfig {
    int dim;
    int populationSize;
    double lower;
    double upper;
    double crossoverRate;
    double mutationProb;
    double mutationSigma;
    int generations;
};

std::vector<double> randomVector(std::mt19937& rng, int dim, double lower, double upper) {
    std::uniform_real_distribution<double> dist(lower, upper);
    std::vector<double> v(dim);
    for (int i = 0; i < dim; ++i) v[i] = dist(rng);
    return v;
}

Individual createIndividual(std::mt19937& rng, const GAConfig& cfg, FitnessFunc f) {
    Individual ind;
    ind.genes = randomVector(rng, cfg.dim, cfg.lower, cfg.upper);
    ind.fitness = f(ind.genes);
    return ind;
}

Individual tournamentSelect(const std::vector<Individual>& pop, std::mt19937& rng, int k=3) {
    std::uniform_int_distribution<int> dist(0, pop.size()-1);
    int bestIdx = dist(rng);
    for (int i=1;i<k;i++) {
        int idx = dist(rng);
        if (pop[idx].fitness < pop[bestIdx].fitness) bestIdx = idx; // minimize
    }
    return pop[bestIdx];
}

std::pair<Individual, Individual> onePointCrossover(const Individual& p1, const Individual& p2, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(1, p1.genes.size()-1);
    int cp = dist(rng);
    Individual c1 = p1; Individual c2 = p2;
    for (size_t i=cp;i<p1.genes.size();++i) {
        c1.genes[i] = p2.genes[i];
        c2.genes[i] = p1.genes[i];
    }
    return {c1,c2};
}

void mutate(Individual& ind, std::mt19937& rng, const GAConfig& cfg) {
    std::uniform_real_distribution<double> prob(0.0,1.0);
    std::normal_distribution<double> gauss(0.0, cfg.mutationSigma);
    for (double& g : ind.genes) {
        if (prob(rng) < cfg.mutationProb) {
            g += gauss(rng);
            if (g < cfg.lower) g = cfg.lower;
            if (g > cfg.upper) g = cfg.upper;
        }
    }
}

void runGA(const std::string& name, FitnessFunc f, const GAConfig& cfg, unsigned seed=42) {
    std::mt19937 rng(seed);
    std::vector<Individual> population(cfg.populationSize);
    for (int i=0;i<cfg.populationSize;++i) population[i]=createIndividual(rng,cfg,f);

    for (int gen=0; gen<cfg.generations; ++gen) {
        std::vector<Individual> next;
        while (next.size() < population.size()) {
            Individual parent1 = tournamentSelect(population,rng);
            Individual parent2 = tournamentSelect(population,rng);
            std::pair<Individual,Individual> children{parent1,parent2};
            std::uniform_real_distribution<double> prob(0.0,1.0);
            if (prob(rng) < cfg.crossoverRate) {
                children = onePointCrossover(parent1,parent2,rng);
            }
            mutate(children.first,rng,cfg);
            mutate(children.second,rng,cfg);
            children.first.fitness = f(children.first.genes);
            children.second.fitness = f(children.second.genes);
            next.push_back(children.first);
            if (next.size() < population.size()) next.push_back(children.second);
        }
        population = std::move(next);

        auto bestIt = std::min_element(population.begin(), population.end(), [](const Individual& a,const Individual& b){return a.fitness<b.fitness;});
        std::cout << name << " Generation " << gen << " best=" << bestIt->fitness << std::endl;
    }
}

int main(){
    GAConfig rastriginCfg{10, 50, -5.12, 5.12, 0.8, 0.1, 0.1, 50};
    GAConfig ackleyCfg{10, 50, -5.0, 5.0, 0.8, 0.1, 0.1, 50};
    GAConfig schwefelCfg{10, 50, -500.0, 500.0, 0.8, 0.1, 20.0, 50};

    runGA("Rastrigin", rastrigin, rastriginCfg);
    runGA("Ackley", ackley, ackleyCfg);
    runGA("Schwefel", schwefel, schwefelCfg);
    return 0;
}

