//
// Created by Xiaoyu Wei on 16/1/2017.
//

/**
 * Initial values for three phase parameters:
 *      - [ 0 ]: the solid
 *      - [ 1 ]: the gas (outside bubble)
 *      - [ 2 ]: the liquid (inside bubble)
 */

/**
 * Assuming computational domain is [0,1]^dim.
 */

#ifndef MBOX_INITIAL_DROP_ON_ROUGH_SURFACE
#define MBOX_INITIAL_DROP_ON_ROUGH_SURFACE

#include <deal.II/base/function.h>
#include "config.h"

namespace mbox {

using namespace dealii;

template <int dim>
class InitialValues0 : public Function<dim> {
 public:
  InitialValues0() : Function<dim>(/*n_component = */ 1){};

  virtual double value(const Point<dim>&  p,
                       const unsigned int component = 0) const;
};

template class InitialValues0<1>;

template class InitialValues0<2>;

template class InitialValues0<3>;

template <int dim>
class InitialValues1 : public Function<dim> {
 public:
  InitialValues1() : Function<dim>(/*n_component = */ 1){};

  virtual double value(const Point<dim>&  p,
                       const unsigned int component = 0) const;
};

template class InitialValues1<1>;

template class InitialValues1<2>;

template class InitialValues1<3>;

template <int dim>
class InitialValues2 : public Function<dim> {
 public:
  InitialValues2() : Function<dim>(/*n_component = */ 1){};

  virtual double value(const Point<dim>&  p,
                       const unsigned int component = 0) const;
};

template class InitialValues2<1>;

template class InitialValues2<2>;

template class InitialValues2<3>;


template <int dim>
double InitialValues0<dim>::value(const Point<dim>&  p,
                                  const unsigned int c) const {
  AssertThrow(c == 0, ExcNotImplemented());

  const double N = 8.;
  const double h = 0.05;

  double height = 0.0;
  for (unsigned int d = 0; d < dim - 1; d++) {
    height *= -sin(2. * M_PI * p[d] * N);
  }

  height = (height > 0.) ? 0.2 + h : 0.2 - h;
  return (p[dim - 1] < height ? 1.0 : 0.0);
}


template <int dim>
double InitialValues1<dim>::value(const Point<dim>&  p,
                                  const unsigned int c) const {
  AssertThrow(c == 0, ExcNotImplemented());

  std::vector<double> R;
  R.push_back(0.35);
  R.push_back(0.35);
  R.push_back(0.35);

  double weighted_distance = 0.0;
  for (unsigned int d = 0; d < dim; d++) {
    weighted_distance += (p[d] - 0.5) * (p[d] - 0.5) / R[d] / R[d];
  }

  return (weighted_distance > 1.0 && InitialValues0<dim>::value(p, c) < 0.50
              ? 1.0
              : 0.0);
}

template <int dim>
double InitialValues2<dim>::value(const Point<dim>&  p,
                                  const unsigned int c) const {
  AssertThrow(c == 0, ExcNotImplemented());

  std::vector<double> R;
  R.push_back(0.25);
  R.push_back(0.25);
  R.push_back(0.25);

  double weighted_distance = 0.0;
  for (unsigned int d = 0; d < dim; d++) {
    weighted_distance += (p[d] - 0.5) * (p[d] - 0.5) / R[d] / R[d];
  }

  return (weighted_distance <= 1.0 && InitialValues0<dim>::values(p, c) < 0.50
              ? 1.0
              : 0.0);
}
}

#endif    // MBOX_INITIAL_DROP_ON_ROUGH_SURFACE_H
