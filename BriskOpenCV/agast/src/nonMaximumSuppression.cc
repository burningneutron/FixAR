/*
    nonMaximumSuppression - this nonMaximumSuppression is a re-implementation
                            of the NMS as proposed by Rosten. However, this
                            implementation is complete and about 40% faster
                            than the OpenCV-version

    Copyright (c) 2010, Elmar Mair
    All rights reserved.
   
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
       * Neither the name of the owner nor the names of its contributors may be 
         used to endorse or promote products derived from this software without 
         specific prior written permission.
   
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdlib.h>
#include "agast/cvWrapper.h"
#include "agast/AstDetector.h"

using namespace std;
using namespace agast;

void AstDetector::nonMaximumSuppression(const std::vector<CvPoint>& corners_all,
		std::vector<CvPoint>& corners_nms)
{
	int currCorner_ind;
	int lastRow=0, next_lastRow=0;
	vector<CvPoint>::const_iterator currCorner;
	int lastRowCorner_ind=0, next_lastRowCorner_ind=0;
	vector<int>::iterator nmsFlags_p;
	vector<CvPoint>::iterator currCorner_nms;
	int j;
	int numCorners_all=corners_all.size();
	int nMaxCorners=corners_nms.capacity();

	currCorner=corners_all.begin();

	if(numCorners_all > nMaxCorners)
	{
		if(nMaxCorners==0)
		{
			nMaxCorners=512 > numCorners_all ? 512 : numCorners_all;
			corners_nms.reserve(nMaxCorners);
			nmsFlags.reserve(nMaxCorners);
		}
		else
		{
			nMaxCorners *=2;
			if(numCorners_all > nMaxCorners)
				nMaxCorners = numCorners_all;
			corners_nms.reserve(nMaxCorners);
			nmsFlags.reserve(nMaxCorners);
		}
	}
	corners_nms.resize(numCorners_all);
	nmsFlags.resize(numCorners_all);

	nmsFlags_p=nmsFlags.begin();
	currCorner_nms=corners_nms.begin();

	//set all flags to MAXIMUM
	for(j=numCorners_all; j>0; j--)
		*nmsFlags_p++=-1;
	nmsFlags_p=nmsFlags.begin();

	for(currCorner_ind=0; currCorner_ind<numCorners_all; currCorner_ind++)
	{
		int t;

		//check above
		if(lastRow+1 < currCorner->y)
		{
			lastRow=next_lastRow;
			lastRowCorner_ind=next_lastRowCorner_ind;
		}
		if(next_lastRow!=currCorner->y)
		{
			next_lastRow=currCorner->y;
			next_lastRowCorner_ind=currCorner_ind;
		}
		if(lastRow+1==currCorner->y)
		{
			//find the corner above the current one
			while((corners_all[lastRowCorner_ind].x < currCorner->x) && (corners_all[lastRowCorner_ind].y == lastRow))
				lastRowCorner_ind++;

			if( (corners_all[lastRowCorner_ind].x == currCorner->x) && (lastRowCorner_ind!=currCorner_ind) )
			{
				int t=lastRowCorner_ind;
				while(nmsFlags[t]!=-1) //find the maximum in this block
					t=nmsFlags[t];

				if( scores[currCorner_ind] < scores[t] )
				{
					nmsFlags[currCorner_ind]=t;
				}
				else
					nmsFlags[t]=currCorner_ind;
			}
		}

		//check left
		t=currCorner_ind-1;
		if( (currCorner_ind!=0) && (corners_all[t].y == currCorner->y) && (corners_all[t].x+1 == currCorner->x) )
		{
			int currCornerMaxAbove_ind=nmsFlags[currCorner_ind];

			while(nmsFlags[t]!=-1) //find the maximum in that area
				t=nmsFlags[t];

			if(currCornerMaxAbove_ind==-1) //no maximum above
			{
				if(t!=currCorner_ind)
				{
					if( scores[currCorner_ind] < scores[t] )
						nmsFlags[currCorner_ind]=t;
					else
						nmsFlags[t]=currCorner_ind;
				}
			}
			else	//maximum above
			{
				if(t!=currCornerMaxAbove_ind)
				{
					if(scores[currCornerMaxAbove_ind] < scores[t])
					{
						nmsFlags[currCornerMaxAbove_ind]=t;
						nmsFlags[currCorner_ind]=t;
					}
					else
					{
						nmsFlags[t]=currCornerMaxAbove_ind;
						nmsFlags[currCorner_ind]=currCornerMaxAbove_ind;
					}
				}
			}
		}

		currCorner++;
	}

	//collecting maximum corners
	corners_nms.resize(0);
	for(currCorner_ind=0; currCorner_ind<numCorners_all; currCorner_ind++)
	{
		if(*nmsFlags_p++ == -1)
			corners_nms.push_back(corners_all[currCorner_ind]);
	}
}

