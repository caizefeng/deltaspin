// Copyright (c) 2016 Florian Wende (flwende@gmail.com)
//
// Distributed under the BSD 2-clause Software License 
// (See accompanying file LICENSE)

#if !defined(FFTLIB_TYPES_HPP)
#define FFTLIB_TYPES_HPP

namespace fftlib
{
	enum elemental_data {BYTE, INT, UINT, C_64, R_64};

	// TODO: 32 bit support
	enum transformation {C2C_64, C2R_64, R2C_64};

	enum scheme {DIRECT = (1<<28), COMPOSED_2D1D = (1<<29), COMPOSED_1D = (1<<30)};
	
	enum backend {FFTW, DFTI};
}

namespace fftlib
{	
	// typedefs for elemental data types depending on
	// the backend chosen for FFT computation.
	template<backend B, elemental_data T>
	class data;

#define FFTLIB_CLASS_DEFINITION(BACKEND, ELEM_TYPE, TYPE) \
	template<> \
	class data<BACKEND, ELEM_TYPE> \
	{ \
	public:	\
		typedef TYPE type_t; \
	}

	FFTLIB_CLASS_DEFINITION(FFTW, BYTE, char);
	FFTLIB_CLASS_DEFINITION(FFTW, INT, int);
	FFTLIB_CLASS_DEFINITION(FFTW, UINT, unsigned);
	FFTLIB_CLASS_DEFINITION(FFTW, C_64, fftw_complex);
	FFTLIB_CLASS_DEFINITION(FFTW, R_64, double);

	// TODO: 32 bit support
	// TODO: definitions for DFTI need to be added here
#undef  FFTLIB_CLASS_DEFINITION	
}

namespace fftlib
{
	template<backend B>
	class trafo_base;

	template<backend B, transformation T>
	class trafo;
	
	template<>
	class trafo_base<FFTW>
	{
	public:
		typedef std::int32_t dim_t;
		typedef fftw_plan plan_t;
	};
       

#define FFTLIB_CLASS_DEFINITION(BACKEND, TRAFO, IN, OUT) \
	template<> \
	class trafo<BACKEND, TRAFO> : public trafo_base<BACKEND> \
	{ \
	public:	\
		typedef data<BACKEND, IN>::type_t in_t;	\
		typedef data<BACKEND, OUT>::type_t out_t; \
	}

	FFTLIB_CLASS_DEFINITION(FFTW, C2C_64, C_64, C_64);
	FFTLIB_CLASS_DEFINITION(FFTW, C2R_64, C_64, R_64);
	FFTLIB_CLASS_DEFINITION(FFTW, R2C_64, R_64, C_64);

	// TODO: 32 bit support
#undef  FFTLIB_CLASS_DEFINITION
}

#endif
