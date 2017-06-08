/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;

double normal_2d_independent_multivariate_pdf(double x1, double m1, double x2, double m2, double s1, double s2)
{
	static const double inv_factor = 1 / (2 * M_PI * s1 * s2);
	double d1 = x1 - m1;
	double d2 = x2 - m2;
	double e1 = d1 * d1 / (2 * s1 * s1);
	double e2 = d2 * d2 / (2 * s2 * s2);

	return inv_factor * std::exp(-(e1 + e2));
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 50;
	particles.resize(num_particles);

	unsigned int seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distX(x, std[0]);
	std::normal_distribution<double> distY(y, std[1]);
	std::normal_distribution<double> distTheta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {

		Particle& p = particles[i];

		p.x = distX(generator);
		p.y = distY(generator);
		p.theta = distTheta(generator);
		p.weight = 1.f;
		p.id = i;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	unsigned int seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distX(0, std_pos[0]);
	std::normal_distribution<double> distY(0, std_pos[1]);
	std::normal_distribution<double> distTheta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {

		Particle& p = particles[i];

		double xd = 0;
		double yd = 0;
		double td = delta_t * yaw_rate;

		if (yaw_rate != 0.0) {
			double mult = velocity / yaw_rate;
			xd = mult * (sin(p.theta + td) - sin(p.theta));
			yd = mult * (cos(p.theta) - cos(p.theta + td));
		} else {
			double mult = delta_t * velocity;
			xd = mult * cos(p.theta);
			yd = mult * sin(p.theta);
		}

		p.x += xd + distX(generator);
		p.y += yd + distY(generator);
		p.theta = p.theta + td + distTheta(generator);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		double best = -1;
		LandmarkObs& obs = observations[i];
		for (LandmarkObs& l : predicted) {
			double d = dist(l.x, l.y, obs.x, obs.y);
			if (best == -1 || d < best) {
				obs.id = l.id;
				best = d;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {

		Particle& p = particles[i];
		std::vector<LandmarkObs> transObservations(observations);

		//translate observations into map coordinates
		for (LandmarkObs& o : transObservations) {
			double nx = o.x * cos(p.theta) - o.y * sin(p.theta) + p.x;
			double ny = o.x * sin(p.theta) + o.y * cos(p.theta) + p.y;
			o.x = nx;
			o.y = ny;
			o.id = -1;
		}

		//save landmarks that are within range
		std::vector<LandmarkObs> landmarks;
		std::map<int, Map::single_landmark_s> hashmap;

		for (Map::single_landmark_s& l : map_landmarks.landmark_list) {
			if (dist(p.x, p.y, l.x_f, l.y_f) < sensor_range) {
				LandmarkObs predLandmark;
				predLandmark.id = l.id_i;
				predLandmark.x = l.x_f;
				predLandmark.y = l.y_f;
				landmarks.push_back(predLandmark);
				hashmap.insert(std::make_pair(l.id_i, l));
			}
		}

		//associate each observation to a landmark
		dataAssociation(landmarks, transObservations);

		//record associations
		std::vector<int> ids;
		std::vector<double> sense_x;
		std::vector<double> sense_y;

		for (LandmarkObs& obs : transObservations) {
			if (obs.id != -1) {
				ids.push_back(obs.id);
				sense_x.push_back(obs.x);
				sense_y.push_back(obs.y);
			}
		}

		p = SetAssociations(p, ids, sense_x, sense_y);

		//adjust weight via multivariate gaussian distribution
		// - the noise variances are assumed to be independent, therefore the covariance would have only diagonal non-zero
		// - elements, which would result in a simplified equation that is the product of independent gaussians for each
		// - dimension

		double weight = 1;
		for (LandmarkObs& measured : transObservations) {
			if (measured.id != -1) {
				Map::single_landmark_s &landmark = hashmap.at(measured.id);

				weight *= normal_2d_independent_multivariate_pdf(measured.x, landmark.x_f,
				                                                 measured.y, landmark.y_f,
				                                                 std_landmark[0], std_landmark[1]);

			} else {
				weight = 0;
			}
		}

		p.weight = weight;

	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	unsigned int seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
	vector<double> weights(num_particles);
	std::vector<Particle> sampled_particles(num_particles);
	std::default_random_engine generator(seed);

	for (int i = 0; i < num_particles; i++) {

		Particle& p = particles[i];
		weights[i] = p.weight;

	}

	discrete_distribution<int> resample_dist(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		sampled_particles[i] = particles[resample_dist(generator)];
		sampled_particles[i].id = i;
	}

	particles = sampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

	return particle;

}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
