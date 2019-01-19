using System;
using System.Collections.Generic;
using System.Text;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Utils
{

    public class Validation
    {

        public static void column_or_1d(y, bool warn = false)
        {
            // """ Ravel column or 1d numpy array, else raises an error
            // 
            // Parameters
            // ----------
            // y : array-like
            // 
            //     warn : boolean, default False
            //     To control display of warnings.
            // 
            //     Returns
            // -------
            // y : array
            // 
            // """
            var shape = np.shape(y);
            if (len(shape) == 1)
                return np.ravel(y);

            if (len(shape) == 2 && shape[1] == 1)
            {
                if (warn)
                {
                    Console.WriteLine("A column-vector y was passed when a 1d array was" +
                                      " expected. Please change the shape of y to " +
                                      "(n_samples, ), for example using ravel().");
                }

                return np.ravel(y);
            }

            throw new Exception($"bad input shape {format(shape)}");
        }
        
        private static void _num_samples(x)
        {
            // """Return number of samples in array-like x."""
            if (hasattr(x, "fit") && callable(x.fit))
                // Don't get num_samples from an ensembles length!
                throw new Exception($"Expected sequence or array-like, got estimator {x}");

            if (!hasattr(x, "__len__") && !hasattr(x, "shape"))
                if (hasattr(x, "__array__"))
                    x = np.asarray(x);
            else
                throw new Exception($"Expected sequence or array-like, got {typeof(x)}");

            if (hasattr(x, "shape"))
            {
                if (len(x.shape) == 0)
                    throw new Exception($"Singleton array {x} cannot be considered a valid collection.");
                // Check that shape is returning an integer or default to len
                // Dask dataframes may not return numeric shape[0] value
                if (isinstance(x.shape[0], numbers.Integral))
                    return x.shape[0];
                else
                    return len(x);
            }
            else
                return len(x);
        }

        public static void check_consistent_length(*arrays)
        {
            // """Check that all arrays have consistent first dimensions.
            // 
            // Checks whether all objects in arrays have the same shape or length.
            // 
            //     Parameters
            // ----------
            // *arrays : list or tuple of input objects.
            //     Objects that will be checked for consistent length.
            // """
            var lengths = [_num_samples(X) for X in
            arrays if X is not None]
            uniques = np.unique(lengths)
            if len(uniques) > 1:
            raise
            ValueError("Found input variables with inconsistent numbers of samples: %r" %[int(l) for l in
            lengths])
        }
        
        public static void check_array(array,
                                       bool accept_sparse = false,
                                       bool accept_large_sparse = true,
                                       string dtype = "numeric",
                                       object order= null,
                                       bool copy = false,
                                       bool force_all_finite = true,
                                       bool ensure_2d = true,
                                       bool allow_nd = false,
                                       int ensure_min_samples = 1,
                                       int ensure_min_features = 1,
                                       bool warn_on_dtype = false,
                                       object estimator = null)

        {
            // """Input validation on an array, list, sparse matrix or similar.
            // 
            // By default, the input is checked to be a non-empty 2D array containing
            // only finite values. If the dtype of the array is object, attempt
            // converting to float, raising on failure.
            // 
            // Parameters
            // ----------
            // array : object
            //     Input object to check / convert.
            // 
            // accept_sparse : string, boolean or list/tuple of strings (default=False)
            //     String[s] representing allowed sparse matrix formats, such as 'csc',
            //     'csr', etc. If the input is sparse but not in the allowed format,
            //     it will be converted to the first listed format. True allows the input
            //     to be any format. False means that a sparse matrix input will
            //     raise an error.
            // 
            //     .. deprecated:: 0.19
            //        Passing 'None' to parameter ``accept_sparse`` in methods is
            //        deprecated in version 0.19 "and will be removed in 0.21. Use
            //        ``accept_sparse=False`` instead.
            // 
            // accept_large_sparse : bool (default=True)
            //     If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
            //     accept_sparse, accept_large_sparse=False will cause it to be accepted
            //     only if its indices are stored with a 32-bit dtype.
            // 
            //     .. versionadded:: 0.20
            // 
            // dtype : string, type, list of types or None (default="numeric")
            //     Data type of result. If None, the dtype of the input is preserved.
            //     If "numeric", dtype is preserved unless array.dtype is object.
            //     If dtype is a list of types, conversion on the first type is only
            //     performed if the dtype of the input is not in the list.
            // 
            // order : 'F', 'C' or None (default=None)
            //     Whether an array will be forced to be fortran or c-style.
            //     When order is None (default), then if copy=False, nothing is ensured
            //     about the memory layout of the output array; otherwise (copy=True)
            //     the memory layout of the returned array is kept as close as possible
            //     to the original array.
            // 
            // copy : boolean (default=False)
            //     Whether a forced copy will be triggered. If copy=False, a copy might
            //     be triggered by a conversion.
            // 
            // force_all_finite : boolean or 'allow-nan', (default=True)
            //     Whether to raise an error on np.inf and np.nan in array. The
            //     possibilities are:
            // 
            //     - True: Force all values of array to be finite.
            //     - False: accept both np.inf and np.nan in array.
            //     - 'allow-nan': accept only np.nan values in array. Values cannot
            //       be infinite.
            // 
            //     .. versionadded:: 0.20
            //        ``force_all_finite`` accepts the string ``'allow-nan'``.
            // 
            // ensure_2d : boolean (default=True)
            //     Whether to raise a value error if array is not 2D.
            // 
            // allow_nd : boolean (default=False)
            //     Whether to allow array.ndim > 2.
            // 
            // ensure_min_samples : int (default=1)
            //     Make sure that the array has a minimum number of samples in its first
            //     axis (rows for a 2D array). Setting to 0 disables this check.
            // 
            // ensure_min_features : int (default=1)
            //     Make sure that the 2D array has some minimum number of features
            //     (columns). The default value of 1 rejects empty datasets.
            //     This check is only enforced when the input data has effectively 2
            //     dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
            //     disables this check.
            // 
            // warn_on_dtype : boolean (default=False)
            //     Raise DataConversionWarning if the dtype of the input data structure
            //     does not match the requested dtype, causing a memory copy.
            // 
            // estimator : str or estimator instance (default=None)
            //     If passed, include the name of the estimator in warning messages.
            // 
            // Returns
            // -------
            // array_converted : object
            //     The converted and validated array.
            // 
            // """
            // accept_sparse 'None' deprecation check
            if (accept_sparse == null)
            {
                Console.WriteLine("Passing 'None' to parameter 'accept_sparse' in methods " +
                                  "check_array and check_X_y is deprecated in version 0.19 " +
                                  "and will be removed in 0.21. Use 'accept_sparse=False' " +
                                  " instead.");
                accept_sparse = false;
            }

            // store reference to original array to check if copy is needed when
            // function returns
            var array_orig = array

            // store whether originally we wanted numeric dtype
            var dtype_numeric = isinstance(dtype, six.string_types) && dtype == "numeric";

            var dtype_orig = getattr(array, "dtype", null);
            if (!hasattr(dtype_orig, "kind"))

                // not a data type (e.g. a column named dtype in a pandas DataFrame)
                dtype_orig = null;

            // check if the object contains several dtypes (typically a pandas
            // DataFrame), and store them. If not, store None.
            var dtypes_orig = null;
            if (hasattr(array, "dtypes") && hasattr(array.dtypes, "__array__"))
                dtypes_orig = np.array(array.dtypes);

            if (dtype_numeric)
                if (dtype_orig != null && dtype_orig.kind == "O")

                    // if input is object, convert to float.
                    dtype = np.float64;
                else
                    dtype = null;

            if (isinstance(dtype, (list, tuple)))
                if (dtype_orig != null && dtype_orig in
            dtype)

            // no dtype conversion required
            dtype = null;
            else

            // dtype conversion required. Let's select the first element of the
            // list of accepted types.
            dtype = dtype[0];

            if (force_all_finite not in (true, false, "allow-nan"))
            throw new Exception("force_all_finite should be a bool or 'allow-nan'" +
                                ". Got {!r} instead'.format(force_all_finite)");

            if (estimator != null)
                if (isinstance(estimator, six.string_types))
                    estimator_name = estimator;
                else
                    estimator_name = estimator.__class__.__name__
            else
                estimator_name = "Estimator";

            var context = " by %s" % estimator != null ? estimator_name : "";

            if (sp.issparse(array))
            {
                _ensure_no_complex_data(array);
                array = _ensure_sparse_format(array, accept_sparse = accept_sparse,
                    dtype = dtype, copy = copy,
                    force_all_finite = force_all_finite,
                    accept_large_sparse = accept_large_sparse);
            }
            else
            {
                // If np.array(..) gives ComplexWarning, then we convert the warning
                // to an error. This is needed because specifying a non complex
                // dtype to the function converts complex to real dtype,
                // thereby passing the test made in the lines following the scope
                // of warnings context manager.
                with warnings.catch_warnings();
                {
                    try
                    {
                        warnings.simplefilter("error", ComplexWarning);
                        array = np.asarray(array, dtype = dtype, order = order);
                    }
                    catch (Exception)
                    {
                        throw new Exception("Complex data not supported\n" + "{}\n".format(array));
                    }
                }

                // It is possible that the np.array(..) gave no warning. This happens
                // when no dtype conversion happened, for example dtype = None. The
                // result is that np.array(..) produces an array of complex dtype
                // and we need to catch and raise exception for such cases.
                _ensure_no_complex_data(array);

                if (ensure_2d)
                {
                    // If input is scalar raise error
                    if (array.ndim == 0)
                        throw new Exception(
                            "Expected 2D array, got scalar array instead:\narray={}.\n" +
                            "Reshape your data either using array.reshape(-1, 1) if " +
                            "your data has a single feature or array.reshape(1, -1) " +
                            "if it contains a single sample.".format(array));

                    // If input is 1D raise error
                    if (array.ndim == 1)
                        throw new Exception(
                            "Expected 2D array, got 1D array instead:\narray={}.\n" +
                            "Reshape your data either using array.reshape(-1, 1) if " +
                            "your data has a single feature or array.reshape(1, -1) " +
                            "if it contains a single sample.".format(array));
                }

                // in the future np.flexible dtypes will be handled like object dtypes
                if (dtype_numeric && np.issubdtype(array.dtype, np.flexible))
                    Console.WriteLine(
                        "Beginning in version 0.22, arrays of bytes/strings will be " +
                        "converted to decimal numbers if dtype='numeric'. " +
                        "It is recommended that you convert the array to " +
                        "a float dtype before using it in scikit-learn, " +
                        "for example by using " +
                        "your_array = your_array.astype(np.float64).");

                // make sure we actually converted to numeric:
                if (dtype_numeric && array.dtype.kind == "O")
                    array = array.astype(np.float64);
                if (not allow_nd && array.ndim >= 3)
                throw new Exception($"Found array with dim {array.ndim}. {estimator_nam} expected <= 2.");
                if (force_all_finite)
                    _assert_all_finite(array, allow_nan = force_all_finite == "allow-nan");
            }

            shape_repr = _shape_repr(array.shape);
            if (ensure_min_samples > 0)
            {
                n_samples = _num_samples(array);
                if (n_samples < ensure_min_samples)
                    throw new Exception($"Found array with {n_samples} sample(s) (shape={shape_repr}) while a" +
                                        " minimum of {ensure_min_samples} is required{context}.");
            }

            if (ensure_min_features > 0 && array.ndim == 2)
            {
                n_features = array.shape[1];
                if (n_features < ensure_min_features)
                    throw new Exception($"Found array with {n_features} feature(s) (shape={shape_repr}) while" +
                                        " a minimum of {ensure_min_features} is required{context}.");
            }

            if (warn_on_dtype && dtype_orig != null && array.dtype != dtype_orig)
            {
                var msg = $"Data with input dtype {dtype_orig} was converted to {array.dtype}{context}.";
                Console.WriteLine(msg);
            }

            if (copy && np.may_share_memory(array, array_orig))
                array = np.array(array, dtype = dtype, order = order);

            if (warn_on_dtype && dtypes_orig != null && {
                array.dtype
            } != set(dtypes_orig))
            {
                // if there was at the beginning some other types than the final one
                //  (for instance in a DataFrame that can contain several dtypes) then
                //  some data must have been converted
                var msg = (&"Data with input dtype {', '.join(map(str, sorted(set(dtypes_orig)))} were all converted to {array.dtype}{context}.";
                Console.WriteLine(msg);
            }

            return array;
        }
        
        public static check_X_y(X,
                                y,
                                bool accept_sparse = false,
                                bool accept_large_sparse= true,
                                string dtype= "numeric",
                                object order= null,
                                bool copy= false,
                                bool force_all_finite= true,
                                bool ensure_2d= true,
                                bool allow_nd= false,
                                bool multi_output= false,
                                int ensure_min_samples= 1, 
                                int ensure_min_features= 1,
                                bool y_numeric= false,
                                bool warn_on_dtype= false, 
                                object estimator= null)
        {
            // """Input validation for standard estimators.
            // 
            // Checks X and y for consistent length, enforces X to be 2D and y 1D. By
            // default, X is checked to be non-empty and containing only finite values.
            // Standard input checks are also applied to y, such as checking that y
            // does not have np.nan or np.inf targets. For multi-label y, set
            // multi_output = True to allow 2D and sparse y.If the dtype of X is
            // object, attempt converting to float, raising on failure.
            // 
            // Parameters
            // ----------
            // X : nd-array, list or sparse matrix
            //     Input data.
            // 
            // y : nd-array, list or sparse matrix
            //     Labels.
            // 
            // accept_sparse : string, boolean or list of string (default=False)
            //     String[s] representing allowed sparse matrix formats, such as 'csc',
            //     'csr', etc. If the input is sparse but not in the allowed format,
            //     it will be converted to the first listed format.True allows the input
            //     to be any format.False means that a sparse matrix input will
            //     raise an error.
            // 
            //     .. deprecated:: 0.19
            //        Passing 'None' to parameter ``accept_sparse`` in methods is
            //        deprecated in version 0.19 "and will be removed in 0.21. Use
            //        ``accept_sparse= False`` instead.
            // 
            // accept_large_sparse : bool (default=True)
            //     If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
            //     accept_sparse, accept_large_sparse will cause it to be accepted only
            //     if its indices are stored with a 32-bit dtype.
            // 
            //     ..versionadded:: 0.20
            // 
            // dtype : string, type, list of types or None (default="numeric")
            //     Data type of result.If None, the dtype of the input is preserved.
            //     If "numeric", dtype is preserved unless array.dtype is object.
            //     If dtype is a list of types, conversion on the first type is only
            //     performed if the dtype of the input is not in the list.
            // 
            // order : 'F', 'C' or None (default=None)
            //     Whether an array will be forced to be fortran or c-style.
            // 
            // copy : boolean(default=False)
            //     Whether a forced copy will be triggered.If copy = False, a copy might
            //     be triggered by a conversion.
            // 
            // force_all_finite : boolean or 'allow-nan', (default = True)
            //     Whether to raise an error on np.inf and np.nan in X.This parameter
            //     does not influence whether y can have np.inf or np.nan values.
            //     The possibilities are:
            // 
            //     - True: Force all values of X to be finite.
            //     - False: accept both np.inf and np.nan in X.
            //     - 'allow-nan': accept only np.nan values in X.Values cannot be
            //       infinite.
            // 
            //     .. versionadded:: 0.20
            //        ``force_all_finite`` accepts the string ``'allow-nan'``.
            // 
            // ensure_2d : boolean (default=True)
            //     Whether to raise a value error if X is not 2D.
            // 
            // allow_nd : boolean(default=False)
            //     Whether to allow X.ndim > 2.
            // 
            // multi_output : boolean (default=False)
            //     Whether to allow 2D y(array or sparse matrix). If false, y will be
            //     validated as a vector.y cannot have np.nan or np.inf values if
            //     multi_output= True.
            // 
            // ensure_min_samples : int (default=1)
            //     Make sure that X has a minimum number of samples in its first
            //     axis(rows for a 2D array).
            // 
            // ensure_min_features : int (default=1)
            //     Make sure that the 2D array has some minimum number of features
            //     (columns). The default value of 1 rejects empty datasets.
            //     This check is only enforced when X has effectively 2 dimensions or
            //     is originally 1D and ``ensure_2d`` is True.Setting to 0 disables
            //    this check.
            // 
            // y_numeric : boolean (default=False)
            //     Whether to ensure that y has a numeric type. If dtype of y is object,
            //     it is converted to float64.Should only be used for regression
            //    algorithms.
            // 
            // warn_on_dtype : boolean (default=False)
            //     Raise DataConversionWarning if the dtype of the input data structure
            //     does not match the requested dtype, causing a memory copy.
            // 
            // estimator : str or estimator instance (default=None)
            //     If passed, include the name of the estimator in warning messages.
            // 
            // Returns
            // -------
            // X_converted : object
            //     The converted and validated X.
            // 
            // y_converted : object
            //     The converted and validated y.
            // """
            if (y == null)
                throw new Exception(("y cannot be None");

            X = check_array(X,
                            accept_sparse = accept_sparse,
                            accept_large_sparse = accept_large_sparse,
                            dtype = dtype, order = order, copy = copy,
                            force_all_finite = force_all_finite,
                            ensure_2d = ensure_2d, allow_nd = allow_nd,
                            ensure_min_samples = ensure_min_samples,
                            ensure_min_features = ensure_min_features,
                            warn_on_dtype = warn_on_dtype,
                            estimator = estimator);
            if (multi_output)
                y = check_array(y, "csr", force_all_finite = true, ensure_2d = false, dtype = null);
            else
            {
                y = column_or_1d(y, warn = true);
                _assert_all_finite(y);
            }

            if (y_numeric && y.dtype.kind == "O")
                y = y.astype(np.float64)

            check_consistent_length(X, y);

            return X, y;
        }

    }

}

}
