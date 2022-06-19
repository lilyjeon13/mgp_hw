#ifndef _BACK_UP_STRUCT_H_
#define _BACK_UP_STRUCT_H_



    namespace Algorithms
    {
//        namespace Sequential
//        {
            enum BackDirection
            {
                stop, //if H == 0 in (Smith Waterman)
                up,
                left,
                crosswise
            };

            struct BackUpStruct
            {
                BackDirection backDirection;
                bool continueUp;
                bool continueLeft;
            };
//        }
    }

#endif
